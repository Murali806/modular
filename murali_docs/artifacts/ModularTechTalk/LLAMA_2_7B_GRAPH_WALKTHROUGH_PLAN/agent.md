# Task rules: MAX Serve / Llama 2 7B walkthrough

These notes capture the user's requirements for this walkthrough. They apply
under this folder and supplement the repository-wide guidance in
`murali_docs/AGENTS.md` and the repository's `AGENTS.md`.

## Goal and technical scope

- Build a source-grounded explanation anchored on
  `max serve --model meta-llama/Llama-2-7b-hf`, with the standard API/worker
  route, one device, and MAX's built-in Graph API unless the user changes the
  scope.
- Explain how the startup path leads into model construction, graph tracing,
  compilation/initialization, and serving. Keep startup, model configuration,
  weight loading, graph work, and HTTP serving as distinguishable phases.
- Treat repository code at the recorded revision as the authority for MAX
  behavior. A source walkthrough is not evidence that the model was run or a
  graph was compiled on hardware.
- Keep Llama 2's Hugging Face model ID
  (`meta-llama/Llama-2-7b-hf`), Hugging Face architecture names such as
  `LlamaForCausalLM`, MAX module paths such as `llama3`, and MAX config types
  such as `Llama3Config` distinct. Explain why names differ when that is part
  of the question.

## Evidence and source links

- Cite the exact source file, symbol, call site, and line for technical claims.
  Link call-tree functions and call sites to the Source Explorer so clicking
  opens the relevant source and shows the complete function when available.
- When the user asks for exact code, show the complete source definition
  unchanged; place explanations alongside it as annotations or separate text.
- Pin generated source snapshots to a repository revision and ensure the
  inspected working-tree source matches that revision. Mark source-derived
  explanations and diagrams `SOURCE`; use the evidence labels from
  `murali_docs/AGENTS.md` for other evidence types.
- Distinguish Hugging Face configuration/tokenizer/generation metadata from
  checkpoint weight files, and cache hits from network downloads.
- Distinguish the in-process `PIPELINE_REGISTRY` / shared `ARCH_LOOKUP` and
  pending lazy import recipes from model objects, downloaded weights, or a
  database on disk. `register_all_models()` registers the supported catalog;
  the `--model` argument selects a model at runtime and does not create a
  model-specific architecture registration.
- Explain `AutoConfig.from_pretrained()`, `AutoTokenizer.from_pretrained()`,
  `GenerationConfig.from_pretrained()`, `download_weight_files()`,
  `register_all_models()`, and `Llama3Config.initialize()` / `finalize()` at
  their actual call sites and lifecycle points when those paths are covered.
- Keep `ModelManifest` role names (for example, `main`, `text_encoder`,
  `unet`, `vae`) distinct from architecture class names. Explain the
  single-model `main` entry, component manifests, Diffusers pipeline indexes,
  and non-component metadata with source-backed examples.
- Explain the graph lifecycle from source and keep `Llama3Config.initialize()`
  / `finalize()` distinct from graph tracing, graph finalization, compilation,
  and runtime initialization wherever the source exposes separate steps.

## Guide layout and interaction

- Use the QnA numbering hierarchy `QnA 1`, `QnA 1.1`, `QnA 1.1.1`, and deeper
  dotted levels. Put each answer beside the question and nest related QnAs in
  expandable/collapsible cells.
- Keep the main HTML dark grey, responsive, readable, and color-coded. Provide
  a collapsible left sidebar, expandable startup map, and expandable code/QnA
  cells. Keep direct-file and offline reading usable.
- Keep QnA 1.1.1–1.1.6 in one continuous, text-only call-stack tree. Put the
  explanation in the tree beside each step; use clear arrows, indentation,
  line spacing, named objects, ownership, data handoffs, and component roles.
- When spacing separates rows within a tree branch, keep its vertical
  connector visible on the spacer row (for example, `|` in the same column).
  This keeps the branch continuous around explanatory text and links; reserve
  fully blank rows for separators outside a connected branch.
- Include the CLI command examples as literal text, including both
  `max serve --model meta-llama/Llama-2-7b-hf --task text_generation` and
  `max serve --model meta-llama/Llama-2-7b-hf`. Explain the difference in
  behavior where relevant. Use text examples, not radio buttons.
- Explain the full call path requested by the user, including task inference
  only when the effective task is `UNDEFINED`, argument normalization,
  `PipelineConfig.from_args()`, registry lookup/materialization, FastAPI
  settings/config/server objects, worker startup, tokenizer and generation
  config loading, weight resolution/download, model config initialization and
  finalization, and graph construction/compilation. Keep each claim tied to
  the correct source call site.
- Put supplementary self-contained HTML pages in `link_htmls/`, give each a
  descriptive filename, use the same readable color coding, and link each
  page from the relevant call-tree explanation with blank spacer lines when
  requested. A call-stack link launched from QnA 1.1 should open the separate
  startup-tree reader.
- Use diagrams and arrows when they clarify object ownership, process
  boundaries, or data flow. Keep labels compact and include a portable text
  equivalent beside important diagrams.

## Authoring and validation workflow

- Keep files created for this walkthrough under this task folder. Treat
  `src/` as the source of truth for generated pages; do not hand-edit generated
  output that the builder will overwrite.
- After editing call-tree or page sources, rebuild from the repository root:

  ```sh
  PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/build.py
  ```

- Validate generated links and source references. Do not run model inference,
  GPU measurements, or benchmarks for documentation-only work; use only the
  checks needed to build and inspect the artifact unless the user asks for
  additional tests or runtime verification.
- Preserve existing Git checkpoints, tags, unrelated modifications, and
  untracked artifacts. Never use a hard reset to roll back. Stage only this
  task's files; follow the repository's signed-off commit and push procedure
  after the requested work is complete and verified.
- Keep the user informed with short progress updates during longer work, and
  finish the requested artifact instead of stopping at a plan when the user
  has asked to proceed.

## Skills

- Documentation-only work uses repository source and the local artifact
  builder; serving, benchmarking, profiling, model-import, and Mojo skills
  are not needed unless the user asks for those tasks.
- If the user asks to serve MAX, use `serve-model`. Use `benchmark-model` for
  latency/throughput measurements, `profile-model` for runtime bottlenecks,
  `debug-model` for silent output-parity failures, and `import-model` when
  adding an architecture from a Hugging Face model ID.
- If writing Mojo, use `mojo-syntax`; add `mojo-gpu-fundamentals` for GPU
  kernels and `mojo-python-interop` for Python/Mojo interop. Use
  `migrate-max-v2-to-v3` only for an explicit V2-to-V3 migration request.
