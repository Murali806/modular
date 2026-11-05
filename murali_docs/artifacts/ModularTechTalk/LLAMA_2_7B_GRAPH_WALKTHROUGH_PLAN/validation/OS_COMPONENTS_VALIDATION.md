# MAX Serve OS-component reader: validation

Date: 2026-11-05, Asia/Colombo. Machine-readable reports record UTC timestamps.

Artifact: [focused OS-component page](../link_htmls/max_serve_os_components_and_flow.html).
Scope: four request/admission/scheduler stories, three teaching state machines,
and a reference atlas for one Linux + NVIDIA serving instance.

Source revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
Model context: `meta-llama/Llama-2-7b-hf`, single-device Graph API.
Device execution: none. `CPU-RUN` describes source/document and headless
browser checks. Animation expresses event order, not measured device timing.

## Environment and commands

Python 3.10.12; Node 22.11.0; Playwright 1.64.1; Chromium 156.0.8078.4.
These commands run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/build.py
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/validate.py
NODE_PATH=/tmp/max-os-playwright/node_modules node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_os_components_browser.cjs
git diff --check
```

The temporary Playwright path belongs to the validation host. The generated
HTML embeds its CSS, JavaScript and full source files; it works directly from
disk. Builds reuse saved reference extracts and source snapshots.

## Source and scope checks — PASS

- 37 source targets across 17 files match the pinned source byte for byte.
  Complete definition bounds, invocation lines, source hashes and original
  copying data are checked.
- 15 component roles identify ownership, layer, interactions and the matching
  call-stack entry.
- Three actor-complete animated flows contain 26 numbered handoffs: startup,
  one request/response and device-driver/buffer interaction.
- Three state machines contain 19 teaching states and eight valid paths.
  Every edge names existing states; every selected path follows valid edges.
- Startup remains in the HTTP-not-accepting state through worker initialization,
  readiness observation and IPC connection; only the final serving handoff
  enables HTTP handling.
- The page reuses 13 reviewed OS/device references with checked extract hashes.
  Local links, IDs, Python source syntax and offline assets pass.
- Four small request stories precede the state machines and reference atlas.
  Their small capacities and event durations are illustrative. The separate
  systems page contains the replica story and advanced lab.

Results: [source_checks.json](source_checks.json) and
[os_components_manifest.json](../os_components_manifest.json).

## Browser checks — PASS

11 grouped checks passed, including all 15 component cards, 26 flow-step
cards, 19 state cards, eight paths and 37 source targets. Seventeen rendered
contrast states passed; zero browser errors or external runtime requests
were observed.

- Direct `file://` loading and no autoplay.
- Persistent hover, focus and touch explanations; Hold/Release and keyboard
  Space/Tab/Escape; exact focus return after the source dialog's close event.
- Correct startup readiness gates and selected event arrows.
- Server startup/failure/shutdown paths; request decode, rejection and safe
  cancellation paths; OS-thread wait/dispatch/preemption paths.
- Full-file source fidelity, selected invocation/definition lines, complete
  copying and source search.
- Native moving SVG markers, explicit Play/Pause/Step/Replay and reduced-motion
  operation.
- Touch cards, source dialogs and state controls at 390×844 and 320×844,
  with diagram scrolling contained inside the page.
- Readable no-JavaScript component and portable sequence/state explanations.
- Print expansion of all explanations and restoration of disclosure states.
- Navigation from the call-stack header to the focused reader.

Normal text meets the inspected 4.5:1 threshold, large text 3:1, and inspected
control outlines, focus indicators and diagram lines 3:1. These rendered-color
checks are part of UI verification, not a complete accessibility audit.

Result: [os_components_browser.json](os_components_browser.json).

The seven new request/system stories have additional behavioral and browser
coverage in [SERVE_STORIES_VALIDATION.md](SERVE_STORIES_VALIDATION.md).

Visual review:

- [Overview](os_components_overview.png)
- [Startup flow](os_components_startup.png)
- [Server state machine](os_components_state_machine.png)
- [GPU driver interaction card](os_components_driver_card.png)
- [Touch card at 390px](os_components_card_mobile_390.png)
- [Touch card at 320px](os_components_card_mobile_320.png)

## Interpretation

`SOURCE` identifies MAX calls. `BOUNDARY` identifies Linux, libuv, Python,
ZeroMQ, Uvicorn and NVIDIA mechanisms explained by external references.
State names group those operations for teaching; they are not claimed as
literal MAX enums. Native driver operations and GPU timing remain behind
their implementation boundary.

The many-client section explains ownership and event-driven waiting within
one serving instance. It does not predict throughput, latency or capacity.
