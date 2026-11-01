# Llama 2 7B: graph finalization and compilation in MAX

**Start with [the interactive HTML guide](index.html).** It contains six
chapters, selectable diagrams, and complete expandable code cells on a dark
grey background. **QnA 1**, immediately after section 1's first paragraph,
explains the `llama3` name, Hugging Face checkpoint files and lookup, and
architecture registration timing. The
[Markdown companion](WALKTHROUGH.md) contains the same explanation and source.

| Artifact | What it provides |
|---|---|
| [Interactive guide](index.html) | Concept → diagram → exact code → line-linked explanation → full-file context |
| [Markdown companion](WALKTHROUGH.md) | Complete reading guide with expandable source and portable diagrams |
| [Explanation research](EXPLANATION_RESEARCH.md) | Twelve reviewed references across hardware, framework internals, and LLM engineering; teaching methods selected from them |
| [Revised design plan](DESIGN_PLAN.md) | Chapter structure, code-cell contract, and completed acceptance checklist |
| [Source map](SOURCE_MAP.md) | Compact navigation through repository files and symbols |
| [Source manifest](source_manifest.json) | Revision, SHA-256 hashes, complete code ranges, and annotations |
| [Validation record](validation/VALIDATION.md) | Source-fidelity and browser checks, commands, environment, and limits |

## Why HTML is the primary format

HTML supports source expansion, line highlighting, full-file search, and
selectable graph/compiler states. Markdown provides a searchable, reviewable
companion. PowerPoint is suited to a later condensed presentation; it is less
useful for reading entire functions and following their callers.

The guide uses methods from the Illustrated Transformer, Harvard's Annotated
Transformer, Distill, PyTorch, JAX, NVIDIA, AMD, Google TPU, vLLM, and GPU
optimization articles. The [research review](EXPLANATION_RESEARCH.md) links
all twelve references and explains where each method is applied.

## What is complete

- **Six chapters:** model/config selection; symbolic construction; graph
  finalization; Python/C++ compilation; compiler stages; initialization and reuse.
- **39 expandable source cells**, containing **2,012 original source lines**.
  The cells retain complete definitions or assignments, including comments and
  docstrings. Annotations appear outside the code.
- **36 complete source files**, available in the source explorer and as local
  snapshots under `sources/`.
- Finalization and compiler-stage explorers, prefill/decode input illustrations,
  a native call sequence, and cold/import/cache-hit comparisons.
- Keyboard expansion, source copying, source search, deep links, chapter-level
  expansion, and mobile layout.
- A **Hide sidebar / Show sidebar** toggle in the top-left toolbar. Hiding the
  sidebar widens the guide; the choice is remembered when browser storage is
  available. The toggle also controls chapter navigation on mobile.
- A single expandable **QnA 1** cell containing the three checkpoint/package/
  registry answers, with links to complete original source files. Its
  checkpoint answer contains nested **QnA 1.1** with expandable **QnA 1.1.x**
  call-flow trees for CLI/config loading, lazy architecture registration,
  tokenizer and generation defaults, weight loading, and MAX's `Llama3Config`
  lifecycle. Each tree node includes its explanation and a source link. QnA
  1.1.2 also distinguishes API `Settings` from pipeline settings and clarifies
  when weight downloads occur.
- Dark grey surfaces and checked text/control contrast, including selected
  diagram states, highlighted code, search matches, and a light print layout.

## Source scope

| Item | Value |
|---|---|
| Model family | Llama 2 7B, confirmed by the user |
| Working checkpoint name | `meta-llama/Llama-2-7b-hf`; checkpoint revision not captured |
| Implementation route | Built-in `LlamaForCausalLM`, one device, Graph API |
| Repository revision | `610f7111836d9a950d45ba52ec9b3e0ed589fe3f` |
| Evidence | Repository source and inspected test definitions |
| Runtime status | No Llama inference or GPU measurements performed |

The three main operations are distinct: `Graph.output()` finishes and verifies
the graph's output contract; compilation produces a MEF-backed artifact;
initialization binds weights and executable state. Model configuration
finalization and device-graph finalization have their own explanations.

## Using the guide

1. Open `index.html` in a browser from this folder. It works with a direct
   `file://` URL; the guide has no external runtime dependencies.
2. Select a chapter or a diagram step. Use **Hide sidebar / Show sidebar** in
   the top-left toolbar to change the reading space.
3. Expand a source cell. Select an annotation to highlight its original lines.
4. Use **View whole file** for imports and surrounding context, or **Copy
   complete source** to copy the original definition.
5. Use the Markdown companion when reviewing or searching in an editor.

## Rebuilding

From this task folder, using Python with the `markdown` package available:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 src/build.py
PYTHONDONTWRITEBYTECODE=1 python3 src/validate.py
```

`src/content.md` holds the explanation, and `src/qna.md` holds the three answers
in QnA 1; `src/cells.py` defines the source
selections and authored annotations. The builder extracts source directly
from the checkout and compares it with the pinned revision before generating
both outputs. `src/widgets.py`, `src/style.css`, and `src/app.js` define the
interactive presentation.

Browser validation uses Playwright. The exact command used on this host is
recorded in [the validation record](validation/VALIDATION.md).

## Validation

Source fidelity, complete ranges, local links, all interactive states,
keyboard expansion, exact copying, full-file search, deep links, direct-file
opening, and desktop/mobile layouts passed their checks. The browser reported
no console errors. Screenshots are in `validation/`.

The local MAX Python runfiles environment lacks `max._core`, so the guide
uses accurately labeled source-derived illustrations and inspected tests.
It does not present those illustrations as captured compiler IR or measured
model execution.
