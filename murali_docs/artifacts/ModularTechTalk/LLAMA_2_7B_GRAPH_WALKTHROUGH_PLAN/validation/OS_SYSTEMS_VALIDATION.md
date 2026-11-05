# MAX Serve OS / GPU systems lab: validation

Date: 2026-11-04. Source revision:
`610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

Artifact: [interactive systems lab](../link_htmls/max_serve_os_driver_and_scaling.html).
Model context: `meta-llama/Llama-2-7b-hf`, single-device Graph API.
Linux + NVIDIA mechanisms are explained from saved source and public references.
The simulator and browser checks run on the CPU; no model or device inference
was executed.

## Environment and commands

Python 3.10.12; Node 22.11.0; Playwright 1.64.1; Chromium 156.0.8078.4.
Run from `murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 src/build.py
PYTHONDONTWRITEBYTECODE=1 python3 src/validate.py
node src/check_os_systems_lab.cjs
NODE_PATH=/tmp/max-os-playwright/node_modules node src/check_os_systems_browser.cjs
NODE_PATH=/tmp/max-os-playwright/node_modules node src/check_browser.cjs
NODE_PATH=/tmp/max-os-playwright/node_modules node src/check_initialize_and_limits.cjs
git diff --check
```

The temporary Playwright path belongs to this validation host. Readers can
open the HTML directly from disk; it embeds its CSS, JavaScript and source.
Normal builds use the saved evidence and do not fetch research pages.

## Source and evidence checks

- 58 source targets across 24 complete files, pinned and checked byte for byte.
- 11 complete displayed definitions containing 404 original source lines,
  with definition/caller locations, exact copying and full-file context.
- Six actor-complete sequences, 48 detailed steps and 22 object explanations.
- 17 reference extracts with URLs and hashes. Each source contributes at most
  180 quoted words; complete articles remain available through their links.
- Local links, IDs, evidence hashes, model geometry and offline assets are
  verified by the shared validator.

Machine-readable results: [source_checks.json](source_checks.json) and
[source/evidence manifest](../os_systems_manifest.json).

## Simulator checks — PASS

All 15 meaningful checks passed. They verify token conservation, unique live
client ownership, queue and page bounds, the opening counts, the 1 GiB case,
overload rejection before generation, tokenizer pressure, slow readers,
cancellation at a live-pass boundary, completion-length capping, replica
warmup/failure/replacement, routing under unequal work and bounded histories.

Opening arithmetic:

```text
1,000 clients = 900 idle + 100 unfinished requests
100 tokenized contexts = 96 waiting + 4 GPU prefill members
KV token bytes = 2 × 32 layers × 32 KV heads × 128 dimensions × 2 bytes
               = 512 KiB
128-token page = 64 MiB
4 × 4096-token contexts + null block = 129 × 64 MiB = 8.0625 GiB
usable request pages = 128
512 prompt + 64 effective output tokens = ceil(576 / 128) = 5 reserved pages
four requests reserve 20 pages; their initial prompts occupy 16 pages
1 GiB pool = 16 blocks − 1 null block = 15 usable pages → 3 reservations
```

Result: [os_systems_simulation.json](os_systems_simulation.json).

## Systems browser checks — PASS

14 grouped checks passed, covering every step card, all 22 object cards,
all 11 source cells and 22 rendered contrast states.

- Explicit Play/Step/Pause, animated SVG packet movement and selected frames.
- Persistent hover cards, Hold/Release, keyboard Space/Tab/Escape and focus
  return after opening a linked source file.
- Source selection, complete-file fidelity, exact copying and search.
- Six table-triggered worked examples with no duplicate screen list.
- Live simulator controls, request inspection, overload, memory pressure,
  cancellation, replica addition, failure and replacement.
- Touch cards, source explorer, examples and lab at 390×844 and 320×844.
- Direct `file://` use, zero external runtime requests and zero browser errors.
- No autoplay, reduced-motion operation, readable no-JavaScript fallbacks
  and print expansion of all explanations and source.
- Navigation from the existing call-stack header to the new standalone page.

Checked normal text meets 4.5:1 contrast and large text 3:1; inspected control
outlines, focus indicators and diagram lines meet 3:1. These rendered-color
checks are part of the UI verification, not a complete accessibility audit.

Result: [os_systems_browser.json](os_systems_browser.json).

Visual review:

- [Desktop overview](os_systems_overview.png)
- [Network packet animation](os_systems_network_animation.png)
- [Scaling lab](os_systems_scaling_lab.png)
- [Worked-example card](os_systems_example_card.png)
- [Touch card at 390px](os_systems_card_mobile_390.png)
- [Touch card at 320px](os_systems_card_mobile_320.png)
- [Touch lab at 390px](os_systems_lab_mobile_390.png)
- [Touch lab at 320px](os_systems_lab_mobile_320.png)

## Existing-guide regressions — PASS

The main guide and call-stack suite passed 24 grouped checks and 33 contrast
states. The initialization-sequence and worked-example suites also passed.
Their results are preserved in [os_systems_regressions.json](os_systems_regressions.json)
while older generated screenshots and reports retain their original baseline.

The call-tree browser expectation now accounts for local-document paths that
appear in the portable export but are omitted from clickable display labels.
Model-configuration owner labels also use readable colors in print. The
source tree, text export and memory-planning page remain unchanged.

## Interpretation limits

`SOURCE` labels identify saved MAX code; `BOUNDARY` labels identify Linux,
network and driver mechanisms explained by external references. `CPU-RUN`
applies to the documentation checks and deterministic teaching simulator.

The simulator uses hypothetical service rates and 10 ms ticks. It serializes
modeled GPU work, reserves prompt plus effective output pages conservatively,
uses exact N slots and includes active prefill in M. Real ZeroMQ HWM is
approximate; MAX has its own allocation and preemption policies. Its per-request
output queue is unbounded in this pinned source.

The API waiting gate, routing, two-second replica warmup and Add/Crash/Replace
actions are external teaching policies. Existing requests retain their owner;
failure does not migrate or replay a partially delivered stream. Simulation
charts cannot predict a real device's throughput or latency.
