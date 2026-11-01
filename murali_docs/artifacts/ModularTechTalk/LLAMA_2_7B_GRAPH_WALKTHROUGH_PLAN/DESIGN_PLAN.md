# Revised plan: concepts with complete, expandable MAX source

Model: **Llama 2 7B**. Route: built-in single-device Graph API decoder.
Source revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

The [research review](EXPLANATION_RESEARCH.md) supports an HTML reading guide
with a Markdown companion. Each chapter pairs a concept and diagram with
complete, expandable source functions and a full-file view.

## Six chapters

| Chapter | Main teaching device | Complete source to expose |
|---|---|---|
| 1. Select and configure Llama | System overview, object contract table | Llama registration; model loading, graph selection, and config finalization |
| 2. Record the computation | One Llama block; packed-token and KV input diagram | Llama module construction and tracing; transformer block; Graph constructor |
| 3. Finalize the graph | Six selectable before/after states | Complete `Graph.output()`; context exit; a small existing graph test |
| 4. Cross into the compiler | Actor-complete sequence and payload labels | Compile orchestration, Python session, native session, engine context, frontend |
| 5. Transform representations | Compiler-stage explorer | MO/MAP/MOGG/MGP builders, pipeline execution, Mojo/JIT scheduling, MEF serialization call site |
| 6. Initialize, execute, reuse | Prefill/decode buffer example; cold/warm comparison | Weight initialization, Llama execution, MEF import/cache branches; device-graph finalization distinction |

The core narrative concentrates on finalization and compilation. Other model
variants appear where the source branches, with links to their original file
context. External framework behavior is used only as a teaching analogy.

## Required code-cell behavior

1. A collapsed summary names the function, its purpose, file, original line
   interval, and evidence category.
2. Expansion reveals the entire function, including decorators and docstrings.
3. Numbered explanations highlight the matching lines without rewriting code.
4. Copy returns the original function text, preserving whitespace.
5. Full-file view reveals imports and surrounding implementation; the original
   file can be downloaded from the local snapshot.
6. Deep links open the relevant cell. Keyboard navigation and native
   expand/collapse controls work without pointer-only interactions.
7. A source manifest records the revision, file SHA-256, and extracted ranges;
   validation compares each cell with its original file at that revision.

## Reading experience

```text
Chapter navigation
        |
        v
Concept + contract -> focused visual -> complete source cell -> self-check
                              |                    |
                        choose a step       highlight explanation
                                                   |
                                             full-file context
```

Use consistent colors for graph construction, compilation, and execution.
Provide static/ASCII fallbacks for important diagrams. State illustrations
describe source behavior; performance charts require recorded measurements.

Use dark grey page and panel backgrounds, with contrasting code, labels,
source links, diagram states, and controls. Keep a readable light print style.
Place one collapsed **QnA 1** cell directly after chapter 1's first paragraph;
it contains the three package/checkpoint/registration answers and opens the
matching complete files in the source explorer.
Keep the Hugging Face `AutoConfig` explanation inside its own nested disclosure
within QnA 1, beside the MAX `Llama3Config` mapping it clarifies.

## Deliverables

| Artifact | Purpose |
|---|---|
| `index.html` | Main interactive guide; source and interactions available offline |
| `WALKTHROUGH.md` | Complete six-chapter reading companion with expandable exact code |
| `EXPLANATION_RESEARCH.md` | Reviewed references and the selected teaching methods |
| `DESIGN_PLAN.md` | This design and its acceptance criteria |
| `SOURCE_MAP.md` | Compact route through repository files |
| `sources/` and source manifest | Exact local snapshots for code fidelity and full-file context |
| `src/` | Reproducible guide builder and focused validators |
| `validation/` | Link/source-fidelity/browser results and small UI screenshots |

All artifacts stay under this task folder.

## Build order and acceptance

- [x] Read hardware, framework, and LLM explanation references.
- [x] Compare presentation formats and define the code-cell contract.
- [x] Extract complete functions and snapshot their original files.
- [x] Build the graph-finalization chapter and its code interactions.
- [x] Complete the other five chapters using the same source-checked structure.
- [x] Generate the HTML and Markdown companion from shared content.
- [x] Verify every source snapshot and cell against the recorded revision.
- [x] Verify local links, code-cell expansion, deep links, copy/full-file views,
      compiler-stage navigation, and mobile layout in a browser.
- [x] Update the entry README with artifacts and observed validation results.
- [x] Add QnA 1 beside the opening `llama3` explanation and keep it synchronized
      with the Markdown companion.
- [x] Check the dark grey theme across interactive states, source search,
      keyboard focus, mobile widths, and print colors.
- [x] Make chapter navigation collapsible from the top-left toolbar, reclaim
      its reading space, and verify keyboard/mobile use and saved preference.

## Evidence boundary

`SOURCE` is sufficient for this code explanation; a measured Llama inference
run is a separate experiment. The available Python runfiles environment was
checked, but importing `max.graph.graph` failed because `max._core` is missing.
Therefore the guide must label conceptual state views and inspected tests
accurately, rather than present a simulated IR view as an executed MAX dump.
