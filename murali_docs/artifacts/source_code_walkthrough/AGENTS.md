# Source Code Walkthrough Rule Book

These instructions apply only to files in
`murali_docs/artifacts/source_code_walkthrough/`.

## Purpose

This folder explains MAX Serve source-code flow for a single
OpenAI-compatible text request. The docs should help a reader understand code
ownership boundaries and data transformation without needing to read every
line of implementation first.

## Style

- Prefer visual explanations over long prose.
- Use concise text only where needed to clarify a diagram.
- Use ASCII diagrams, flow charts, decision trees, and side-by-side mappings.
- Keep examples concrete, with realistic request IDs, model names, messages,
  sampling params, and output objects.
- Avoid vague statements such as "does processing" or "handles logic"; name
  the input, action, and output.

## Focused Notes

For each important function or object, create a focused note file named with
the sequence number and symbol, for example:

```text
0001_parse_openai_request_body.md
0002_get_pipeline.md
0003_openai_parse_chat_completion_request.md
0004_TextGenerationRequest.md
0005_streaming_vs_non_streaming.md
0006_tokenizer_new_context.md
```

Use this structure by default:

```text
# `symbol_or_topic`

Sources
One-Line Purpose
Big Picture
Inputs And Output
Worked Example
Sequence View
Decision Tree, if useful
Q&A, if useful
Where It Goes Next
Short Version
```

Not every section is required, but `One-Line Purpose`, `Inputs And Output`,
`Worked Example`, and `Where It Goes Next` should usually be present.

## Worked Examples

Use the same left / middle / right style when explaining transformations:

```text
INPUT                                  ACTION                                  OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
source value                           operation                               result value
  example                            ------------------------------->          example
```

When a branch exists, make the branch explicit:

```text
+------------------------------------------------------------------------------------------------------+
| PROMPT SOURCE IF / ELSE                                                                              |
|------------------------------------------------------------------------------------------------------|
| IF prompt exists                                                                                     |
| ...                                                                                                  |
| ELSE prompt is None                                                                                  |
| ...                                                                                                  |
|------------------------------------------------------------------------------------------------------|
| END IF / ELSE                                                                                        |
+------------------------------------------------------------------------------------------------------+
```

For values produced by an earlier row, do not imply they were original inputs.
Show the flow explicitly:

```text
rendered prompt string  -> tokenize prompt/messages -> produced token_ids

produced token_ids
  |
  | create TokenBuffer
  v
context.tokens = TokenBuffer(...)
```

## Main Walkthrough Links

`0000_source_code_walkthrough.md` is the index / main path. When adding a new
focused note, link it from the relevant bullet using this style:

```html
<strong><em><a href="000N_note_name.md"><span style="color:#0b63ce">See focused note: 000N_note_name.md</span></a></em></strong>.
```

Keep these links on the same bullet as the symbol being explained when
possible.

## Collapsible Sections

- In `0000_source_code_walkthrough.md`, keep one top-level `<details>` section
  per major source file.
- Do not add nested collapsible sections in `0000_source_code_walkthrough.md`.
- In focused notes, Q&A entries may use collapsible `<details>` blocks.
- Check that `<details>` and `</details>` counts match after edits.

## Source Links

Files in this folder are three levels below the workspace root. Link to source
files with paths relative to this folder:

```text
../../../max/...
../../../Mojo/...
../../../docs/...
```

Do not use stale links like:

```text
../../max/...
```

## Accuracy Rules

- Read the referenced source code before explaining a function.
- Explain actual inputs, outputs, branches, and errors from the code.
- If a value is illustrative, label it as illustrative or model-specific.
- Avoid claiming exact token IDs are universal; tokenizer outputs are
  model-specific.
- Distinguish data that already exists from data produced by the current step.

## Validation Before Finishing

Run at least:

```bash
git diff --check -- murali_docs/artifacts/source_code_walkthrough
```

If editing collapsible sections, also verify counts:

```bash
rg -o '<details>' murali_docs/artifacts/source_code_walkthrough | wc -l
rg -o '</details>' murali_docs/artifacts/source_code_walkthrough | wc -l
```

## Git Completion

After completing documentation edits in this folder, commit the changes with an
appropriate descriptive message and push them to origin:

```bash
git push origin main
```
