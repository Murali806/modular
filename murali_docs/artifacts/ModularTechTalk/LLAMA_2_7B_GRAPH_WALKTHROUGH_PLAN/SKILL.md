---
name: max-serve-visual-walkthrough
description: Create or improve source-linked animation HTML in LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN for MAX Serve, InferenceSession, graph compilation, Mojo, MEF, inference and memory ownership. Use when a component or call-stack entry needs an interactive visual walkthrough matching the accepted local animation pages, or when updating their teaching requirements.
---

# MAX / Mojo interactive visual walkthrough

Teach through visible changes to named objects, references, buffers, queues,
resources and execution state. Match the accepted animations' explanatory
power as well as their appearance. A sequence of highlighted boxes and changing
paragraphs does not meet that goal when the source exposes concrete handoffs
and resource changes to draw.

## Start in a fresh session

1. Resolve `TASK` as the directory containing this file and `REPO` as the
   repository root. Paths in the references are relative to `TASK` unless
   stated otherwise. Read [agent.md](agent.md) and applicable `AGENTS.md` files.
2. Take the current request as the scope. A standalone `InferenceSession`
   lesson starts inside that constructor; a MAX/Mojo compiler request follows
   the engine route. The Serve opening sequence and its thread badges apply
   only to the two Serve readers.
3. For animation work, read [the shared authoring recipe](references/animation_authoring.md)
   and select the closest family in [the example catalog](references/animation_examples.md).
   Inspect its rendered page, builder/template, behavioral model, controller,
   and relevant checks. Read the current files; generated HTML or old
   screenshots may predate edits to the authoring source.
4. Resolve the requested symbols and callers against the pinned source. Map
   each teaching decision to a source span before designing the transitions.
   Preserve independent working-tree changes.

The skill remains in this artifact folder. If it is absent from a new
session's available-skills list, request it by path:

```text
Read murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/SKILL.md
and its relevant animation references. Create a link HTML for <symbol/topic>,
matching the motion, state detail and interactions of the cited local examples.
```

## Choose the relevant instructions

| Work | Read / reuse |
| --- | --- |
| Any new or changed animation | [Shared recipe](references/animation_authoring.md); matched files in [example catalog](references/animation_examples.md) |
| Serve admission, API, worker, batch constructor, busy/slow-reader/replica stories | [Serve requirements and invariants](references/max_serve_os_animations.md); Serve family in the catalog |
| Session creation, devices, native context, handles and lifetimes | InferenceSession family in the catalog |
| Graph/dialect transformations, Mojo systems control, target tools/binaries, MEF and driver/buffer movement | [Compiler and systems animations](references/compiler_and_systems_animations.md); compiler workbench family in the catalog |
| Memory budgets, KV pages, input origins or parent/worker planning | Memory family in the catalog; memory rules in [agent.md](agent.md) |
| Main QnA/call-stack guide or requirements-only updates | Relevant sections of [agent.md](agent.md); update the affected reference without rebuilding pages |

## Preserve the accepted experience

- Use a light theme, a full-width scene and compact controls. The compiler
  workbench's drafting surface and physical artifact shapes are an accepted
  variation for compiler/systems concepts. Keep the main guide's dark theme scoped to it.
- Keep a numbered lesson badge inside the animation. Hover/focus reveals a
  temporary explanation card; moving into the card keeps it open, and leaving
  both dismisses it. Support touch and Escape. Keep the circled **L** legend
  button at the right edge of the top bar; it opens on click.
- Give components distinct, stable colors shared by their outlines and legend
  swatches. Legend descriptions identify role, actual owner, object/process/
  thread/artifact/device kind, and relevant IPC or synchronization. Keep
  lifecycle colors and identity labels separate from component colors.
- Draw stable actor identities, occupied/empty slots, real ownership edges,
  decision outcomes and repeatable work where the topic exposes them. Keep
  the current event readable in larger bold text on a fitted tinted backing.
- Keep the state model separate from visual motion. Step/Reset and scenario
  selection must be deterministic. Include Play/Pause without autoplay; use
  Back/seek for reversible timeline lessons. Respect reduced motion and
  pause when hidden, offscreen or inside a closed section.
- Keep source cards, complete unchanged source definitions, an offline Source
  Explorer, portable text equivalents and usable no-JavaScript/print reading.
  Mark illustrative quantities and native boundaries honestly.

## Complete and verify the work

Author in `src/`, integrate with `src/build.py`, and generate descriptive
`link_htmls/` pages. For a requested call-stack supplement, add its link directly
under the relevant ownership/call entry, retaining connector-only spacer rows.
Preserve fragment IDs and existing lessons. Shared Serve lesson edits reach
both readers; a standalone engine lesson does not require Serve page changes.

For page changes, use the family-specific state, browser and source/link checks
in the catalog. Review the rendered animation at intermediate states, not just
its first and last frame. Check that a reader can see what moved, what waited,
what was created or retained, and why a branch succeeded or stopped. Inspect
card dismissal, color matching, dynamic text, routes, narrow-screen layout and
offline behavior. A source illustration does not prove MAX/GPU execution.

For instruction-only changes, validate skill frontmatter, reference paths and
`git diff --check`; page rebuilds and model/browser workloads are unnecessary.
Keep explicit user requirements, accepted design choices, source invariants
and illustrative settings distinguishable when updating these instructions.
Follow the scoped Git workflow in [agent.md](agent.md).
