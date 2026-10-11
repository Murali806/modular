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

## Folder Structure Guide

`0000_A_folder_structure.md` is the folder-structure guide for this workspace.
It explains the repository through annotated ASCII trees rather than separate
prose-only descriptions.

- Keep the complete `max/` tree inside one expandable `<details>` section.
- Give each other root-level folder its own expandable `<details>` section.
- Put exactly one visible tree in each root-folder section. Do not split one
  root folder into separate "more recursive levels" or example trees.
- Continue each tree recursively through the actual leaf folders and files
  that are being documented. Do not stop after one or two levels when deeper
  structure is available.
- Add a concise inline `#` comment for every documented folder and file. The
  comment should explain its responsibility, not merely repeat its name.
- Use ASCII tree characters and stable indentation so the parent-child
  relationship remains readable when rendered as Markdown.
- Preserve the existing `max/` tree when changing non-MAX root-folder trees;
  edit it only when the request explicitly targets MAX.
- Keep the folder guide linked from `0000_source_code_walkthrough.md` using
  the current relative-link and bold-italic focused-note style.
- When the guide is renamed, update every self-reference and walkthrough link;
  do not leave stale references to the previous filename.

Example shape:

````html
<details>
<summary><strong>root-folder/</strong> - responsibility</summary>

```text
root-folder/                         # root responsibility
|
+-- src/                             # implementation source
|   +-- module.py                     # module responsibility
|       +-- helper.py                 # helper responsibility
+-- tests/                            # validation code
    +-- test_module.py                # module tests
```

</details>
````

The guide may cover build, runtime, language, support, tooling, cache,
configuration, documentation, legal, and study-artifact folders. Keep each
root-folder tree focused on ownership and purpose, and avoid inventing files
when the repository does not contain them.

## Complete Code Walkthroughs

When requested, create a separate complete-code walkthrough alongside the
focused note. A complete-code walkthrough contains the actual source code in
execution order, with additional concise comments and visual diagrams that
explain the data flow, branch decisions, variable transformations, and handoff
between functions. Do not replace or rewrite the existing focused note; leave
it as-is and add exactly one extra link to the new complete-code walkthrough.

Use this naming pattern, where the letter identifies the walkthrough section
and the number identifies its sequence within that section:

```text
0001_A_1_parse_openai_request_body_<appropriate_suffix>.md
0001_A_2_parse_openai_request_body_<appropriate_suffix>.md
0001_B_1_parse_openai_request_body_<appropriate_suffix>.md
0001_B_2_parse_openai_request_body_<appropriate_suffix>.md
0001_C_1_parse_openai_request_body_<appropriate_suffix>.md
```

- Use `000N_A_1`, `000N_A_2`, and so on for the first section of note `000N`.
- Start the next section with `000N_B_1`, then continue with `000N_B_2`, etc.
- Continue alphabetically with `000N_C_1`, `000N_C_2`, and later letters when
  more sectionization is needed.
- Keep the suffix specific enough to identify the covered source-code path or
  phase, for example `http_ingress` or `request_validation`.
- Preserve source-code order and include only relevant, verbatim source code;
  do not replace implementation blocks with pseudocode or simplified code.
- Add concise walkthrough-only comments around or inside the copied source to
  explain what each block receives, does, changes, returns, or hands off. Mark
  inserted comments clearly when they are not present in the source file.
- Add visual explanations wherever they make the execution easier to follow.
  Prefer ASCII sequence diagrams, branch/decision flows, call flows, and
  variable/data-flow diagrams that use the actual source-code names.
- Show important variable evolution explicitly, for example:

  ```text
  request: Request
      |
      | await request.body()
      v
  raw: bytes
      |
      | json.loads(raw)
      v
  parsed: object
      |
      | model_cls.model_validate(parsed)
      v
  validated request: _TRequest
  ```

- Every complete-code walkthrough must also include a concrete input/output
  example with fully spelled-out variable values at each important step.
- Do not use opaque values such as `Request(...)`, `Context(...)`, `list[...]`,
  `{...}`, or truncated JSON in a concrete variable-flow example.
- For large framework objects, define a complete logical snapshot containing
  every field used by the code being explained. For Pydantic objects, a full
  `model_dump(exclude_none=True)` value is preferred over an abbreviated repr.
- Keep one consistent example request across related files so a reader can
  follow the same request ID, model, messages, sampling settings, token IDs,
  context, batch, and output through the full lifecycle.
- Label tokenizer IDs, logits, device buffers, and generated tokens as
  illustrative when their exact values are model- or runtime-dependent, but
  still provide the complete illustrative list or array without ellipses.
- Show both the input and output values for branches. If a branch does not
  change a value, repeat the full value or explicitly state `unchanged` after
  the full input value has been defined.

- Keep diagrams adjacent to the source block they explain. Diagrams supplement
  the actual code; they do not replace it.
- When the path spans multiple source files, use `000N_A_1`, `000N_A_2`, and
  later sequence numbers in execution order. Use `B`, `C`, and later letters
  only when a new logical section is needed.
- Link every new complete-code walkthrough from its original focused note and
  keep the link in the established bold-italic focused-note style.
- Keep complete-code walkthrough links relative to the documentation folder,
  and verify that every referenced source path and line anchor is valid.

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
