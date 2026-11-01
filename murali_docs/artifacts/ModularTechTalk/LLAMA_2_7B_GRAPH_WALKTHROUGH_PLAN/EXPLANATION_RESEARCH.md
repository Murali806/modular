# How to explain the MAX graph/compiler path

Research date: **2026-11-01**. Scope: teaching methods in AI hardware,
framework, and LLM engineering material. Twelve substantive references were
read; an additional interactive site received a limited browser check.

**Recommendation: an interactive HTML guide with a Markdown companion.**
Use diagrams to establish the idea, then let the reader expand the complete
MAX function and follow explanations tied to its source lines.

These references inform presentation. MAX implementation claims come from
the local repository at `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
Historical hardware results and other frameworks' behavior are not MAX
measurements or API contracts.

## References and the methods they demonstrate

| Reference inspected | Observed explanation method | Application to this walkthrough |
|---|---|---|
| [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Starts with the whole model, opens its components, then introduces tensor representations and attention operations. The same visual vocabulary persists across scales. | Start with the six-stage lifecycle, then open a Llama block, its symbolic operations, and the graph signature. Keep colors consistent across diagrams. |
| [Harvard NLP — The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | Alternates model concepts with working Python implementations in a notebook-style document. Architecture, code, and examples appear near each other. | Place each concept immediately above a complete expandable MAX function. Keep the original implementation intact and put annotations beside it. |
| [Distill — Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) | Small, labeled visual explanations develop memory, attention, and recurrent computation incrementally. Captions explain what the reader should notice. | Use a focused graph-state diagram for each finalization step; provide a sentence explaining what changed after the selected step. |
| [PyTorch — A Tour of PyTorch Internals, Part I](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/) | Follows Python-visible objects into extension types and native implementation, with named files and concrete code. | Follow `_compile_module()` into the C++ session, engine context, frontend, compiler, and MEF reader. Give each boundary a caller, payload, callee, and result. |
| [PyTorch — Dynamo Deep-Dive](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html) | Builds from a small traced program and its graph to the interpreter machinery, guards, symbolic shapes, and graph breaks. | Separate Python tracing, the recorded graph, and later compilation. Show what is static and what remains a runtime input. Dynamo's guard behavior is not assumed to apply to MAX. |
| [JAX — Just-in-time compilation](https://docs.jax.dev/en/latest/jit-compilation.html) | Places Python examples beside `jaxpr` output, explains tracing side effects, and separates first-call compilation from reuse. | Explain graph construction versus execution with explicit object/state transitions. Give cached compilation its own view. This inspected page is retained by JAX as older documentation. |
| [NVIDIA — Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) | Keeps one small kernel fixed while changing launch/synchronization strategy. CPU/GPU timeline rows make the effect visible; the article states the experimental hardware. | Use an ordered startup/execution sequence and a comparison of cold compilation, artifact reuse, and repeated execution. Explain CUDA capture separately from `Graph.output()`. |
| [Google Cloud — An in-depth look at the first TPU](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu) | Connects a small neural-network calculation to matrix multiplication, then to an architectural block diagram and systolic data movement. | Connect Llama tensor operations to graph/compiler representations before discussing backend kernels. Use ownership/dataflow diagrams at the Python/native boundary. |
| [AMD GPUOpen — AMD matrix cores](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/) | Defines an operation's operands, explains their register/lane layout, and works through concrete intrinsic examples. | Put shape, dtype, device, and meaning next to each graph input. Distinguish symbolic input types from the buffers supplied at execution. |
| [vLLM — PagedAttention introduction](https://vllm.ai/blog/2023-06-20-vllm) | Motivates a memory problem, uses a paging analogy, then shows logical blocks, physical blocks, and sharing. | Show packed tokens, row offsets, and KV inputs together. Explain the role of KV handles without suggesting that the entire cache is rebuilt for every token. |
| [Horace He — Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) | Organizes performance reasoning around compute, bandwidth, and overhead, then relates optimizations to those regimes. | Explain the purposes of fusion, buffer reuse, and reduced launch overhead. Keep the purpose of a pass separate from an unmeasured performance claim. |
| [Simon Boehm — CUDA Matmul Optimization Worklog](https://siboehm.com/articles/22/CUDA-MMM) | Changes one kernel design at a time; uses memory-access diagrams, code, and measured comparisons to explain each change. | Walk through compiler representations in order, showing each stage's input, transformation, and output. For a specific fusion, require actual before/after IR before claiming it happened. |

The references cover hardware architecture, GPU programming, Python/native
framework internals, compiler tracing, LLM memory management, and visual model
explanations. Official guides are included alongside blogs because they are
particularly useful examples of explaining source code and intermediate forms.

## What the browser inspection established

- Distill's article rendered its labeled inline SVG explanations and captions.
- JAX rendered code and corresponding intermediate output. Its HTML contains
  `details` elements, but the inspected elements did not establish a code-folding
  interface. Code folding below is our design choice, not a claimed JAX feature.
- [Brendan Bycroft's LLM Visualization](https://bbycroft.net/llm) loaded a canvas
  in the headless browser, but the full walkthrough interface did not finish
  appearing during the check. Its declared 3D walkthrough is an additional
  design reference, not a fully evaluated interaction here.

The Google TPU article's old `/products/compute/` address returned a soft
not-found page; the working `/products/ai-machine-learning/` address above was
read. The PyTorch stable URL required following two document redirects to the
versioned deep-dive address above. A successful HTTP status alone was not
treated as evidence that the article had been read.

## Choose a method for the question

| Reader's question | Best presentation | Example in the MAX guide |
|---|---|---|
| Where am I in the system? | Overview with progressive detail | Configure → trace → finalize → compile → initialize → execute |
| Who calls whom? | Sequence/call tree | Python session → C++ session → CPU compilation task → frontend |
| What changes at this line? | State before/after plus code annotation | `functionType`, `signature`, `mo.output`, output names |
| What does this buffer represent? | Shape/ownership table and buffer diagram | Packed tokens, row offsets, KV leaves |
| Why does this pass exist? | Input → transformation → output | MO cleanup, MAP fusion, MGP memory planning |
| What happens on another call? | Branch comparison | Cold compile, Python MEF import, native MEF cache hit |
| How does the implementation actually do it? | Complete function with focused annotations | Expand `Graph.output()`; click the signature explanation to locate its lines |
| Where is the surrounding context? | Complete file view | Imports, class definitions, neighboring helpers, and original comments |
| How do I check my understanding? | Short question with a collapsible answer | Does `Graph.__exit__()` compile the graph? |

## Format decision

| Format | Strength for this task | Limitation | Role |
|---|---|---|---|
| **HTML** | Expand/collapse functions, highlight source lines, step through states, compare branches, navigate full files | Requires browser validation and a maintained build source | **Primary guide**; works locally without a network dependency |
| **Markdown** | Easy to search, review, version, copy, and link; supports `<details>` in common renderers | Interactive highlighting and synchronized diagrams are limited; rendering varies | Complete reading/source companion |
| **PowerPoint** | Useful for a spoken presentation with one visual per slide | Full functions are awkward to read; source navigation and code folding are weak | A later condensed presentation, derived from the completed guide |

## Code-cell design

```text
CONCEPT: Finalize the graph's output contract
  One-sentence purpose + input/output contract + focused diagram

> Exact MAX source: Graph.output()                       [expand]
  path | symbol | original lines | source revision | SOURCE
  +-----------------------------------+------------------------+
  | complete, original function       | 1. Flatten output tree |
  | original line numbers             | 2. Add chain results   |
  | selected lines highlighted        | 3. Set signature       |
  |                                   | 4. Add terminator      |
  | [Copy function] [View whole file]  | 5. Verify              |
  +-----------------------------------+------------------------+

> Check your understanding                               [expand]
  Question + answer + source pointer
```

The expanded function includes decorators, comments, docstrings, and every
line of its body. Explanations live outside the source text. A full-file view
provides surrounding context and a downloadable original snapshot. Source
hashes and a revision record make the relationship to the checkout checkable.

Use numbered block explanations where several lines implement one idea.
Line highlights help locate the block; they do not replace the complete
function. Diagrams and examples carry their own evidence labels so an
illustration cannot be mistaken for captured Llama execution or compiler IR.

## Implementation plan

The concrete chapter and build plan is in [DESIGN_PLAN.md](DESIGN_PLAN.md).
The earlier verified source route remains in [SOURCE_MAP.md](SOURCE_MAP.md).
