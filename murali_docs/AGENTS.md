# MAX Internals Course Rules

These rules apply to every file under `murali_docs/`.

## Visual-First Teaching

- Follow `../../visual_diagram_catalog.md`.
- Keep prose minimal. Prefer labels, numbered callouts, and compact tables.
- Put a portable ASCII diagram beside every important Mermaid diagram.
- Use actor-complete sequence diagrams for request and startup flows.
- Use block diagrams for ownership and component boundaries.
- Use state diagrams for request, server, cache, and failure lifecycles.
- Use buffer diagrams for tokens, tensors, KV pages, and host/device movement.
- Split complex subjects into several focused diagrams instead of one dense map.

## Evidence Labels

Mark technical claims and diagrams with one of:

- `CPU-RUN`: executed on this CPU-only host.
- `SOURCE`: proven by repository source.
- `TEST`: proven by a repository test.
- `COMPILE-ONLY`: cross-compiled or inspected without accelerator execution.
- `GPU-LAB`: requires a future GPU host for runtime measurement.
- `BOUNDARY`: implementation is behind a native or external boundary.

Never present `SOURCE` or `COMPILE-ONLY` evidence as measured GPU behavior.

## Artifacts

- Store small Markdown evidence in `murali_docs/artifacts/`.
- Store reusable learning programs in `murali_docs/labs/`.
- Do not commit model weights, MEFs, raw profiler databases, secrets, or large
  benchmark outputs.
- Record exact commands, revision, model ID, device, and result for each run.

## Git Workflow

- Work on `main` unless the user says otherwise.
- Keep commits atomic and sign them with `git commit -s`.
- After verification, push completed work with `git push origin main`.
- Never discard unrelated or user-authored changes.
