# Worker deserialization explanation validation

- Date: 2026-11-05.
- Evidence: `SOURCE` for MAX behavior; `BOUNDARY` for upstream reports;
  `CPU-RUN` for browser checks and the synthetic Python experiment.
- Saved MAX source revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
- Documentation baseline: `36b70c3f4f5e96314076532baf80f5c6cfadca79`.
- QnA/research follow-up baseline: `7a106e54e648cacf6f975c363e6aff9da868f849`.
- Class-identity QnA baseline: `6d980861f181e2b4fda1849429a86039c1cee921`.
- Model context: `meta-llama/Llama-2-7b-hf`; temperature examples are illustrative.
- Device: local CPU browser; no model inference or GPU measurement.

## Build

Regenerated only the startup reader, its exports and its manifest from the
existing reader's saved source payload:

```bash
python3 -B - <<'PY'
from pathlib import Path
import json
import re
import sys

task = Path('murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN').resolve()
src = task / 'src'
sys.path.insert(0, str(src))
from build import python_span
from call_stack import build_call_stack

reader = (task / 'max_serve_llama2_7b_startup_call_stack.html').read_text()
payload = re.search(r'<script id="stack-data" type="application/json">(.*?)</script>', reader, re.S)
assert payload is not None
data = json.loads(payload.group(1))
count = build_call_stack(data, task, src, python_span)
print(f'Rebuilt the startup reader and its text, Markdown and manifest outputs: {count} source targets.')
PY
```

Result: 129 source targets preserved; one explanation link added immediately
after `MAXModelConfig.__setstate__()` in the worker branch.

## Browser verification

```bash
node /tmp/check_worker_deserialization.cjs
git diff --check
```

The temporary Playwright check used the installed browser at
`/tmp/max-os-playwright/node_modules/playwright`. Chromium required execution
outside the filesystem sandbox. The final run passed all twelve check groups:

- Exact link placement, one occurrence and opening in a new tab.
- Visible tree, copy payload, text/Markdown exports and manifest agree.
- Four initially collapsed sibling QnA cells; independent Enter/Space controls.
- Complete original and derived-state answers, diagrams and source excerpts.
- Six remote-code issue types, versioned evidence and the labeled CPU probe.
- The new class-identity explanation, normal/reload diagram controls, and
  Enter/Space activation of both scenarios.
- Local links, fragment targets, source lines and experiment evidence hashes.
- Deep links and cross references reveal the relevant answer and its ancestors;
  malformed hashes preserve the disclosure state without JavaScript errors.
- Collapsed and expanded layouts for both scenarios at 1440px, 390px and
  320px; horizontal
  scrolling stays inside diagrams and tables. Screenshots visually reviewed.
- Print expands all answers, shows both scenario diagrams, and restores the
  prior disclosure state; print colours and keyboard skip navigation pass.
- Existing source explorer still highlights `__setstate__()` at line 1024.
- Native QnA controls and return navigation work directly from `file://` with
  offline mode enabled and JavaScript disabled; both static diagrams remain
  visible when their QnA is opened.

The final contrast check examined 416 samples in the normal scenario and
414 in the reload scenario at each screen width, with a minimum text ratio
of 6.621:1. Print checked 419 samples, with a minimum of 6.233:1.
No contrast failures, JavaScript errors or external resource requests occurred.
Temporary screenshots and the detailed report are in
`/tmp/max-worker-deserialization-checks/`.

## Research and Python experiment

Inspected the saved MAX source and Transformers `v5.6.0`, matching this
checkout's Linux dependency lock. Five primary upstream reports illustrate
specific historical failures; each retains its reported date, library version
and model context. Source URLs, downloaded-source hashes and interpretation
limits are recorded in
[`trust_remote_code_research.json`](../evidence/trust_remote_code_research.json).

```bash
python3 -B murali_docs/labs/worker_config_pickle_demo.py --json-output murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/validation/TRUST_REMOTE_CODE_PICKLE_PROBE.json
```

Result with Python `3.10.12` on CPU:

- Same-process round trip preserves the synthetic configuration class.
- An isolated worker with no importable module raises `ModuleNotFoundError`.
- Replacing the parent module's class makes an older instance raise
  `PicklingError` during serialization.
- Transferring plain source identity and rebuilding in the worker succeeds
  with `hidden_size = 4096`.

The standard-library program creates its own local class and uses fresh
isolated Python subprocesses. It exercises the pickle mechanisms without
loading HF/MAX or executing repository-supplied remote code. These results
are not a reproduction of the reported model-specific bugs. The recorded
output is [`TRUST_REMOTE_CODE_PICKLE_PROBE.json`](TRUST_REMOTE_CODE_PICKLE_PROBE.json).


## Class identity walkthrough

The fourth QnA explains the distinction between a module, a class object, and
an instance; the class-reference/state contents of an ordinary pickle; why a
worker can use its own class object; and why reloading a module can invalidate
an older instance's sender-side class lookup. CPython `v3.10.12` source confirms
the identity comparison in both serializer implementations. The vLLM report's
traceback is identified as V0-engine-to-worker serialization; reload is a
separate demonstrated mechanism, not an asserted root cause for that report.

```bash
python3 -B murali_docs/labs/pickle_class_identity_walkthrough.py --json-output murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/validation/CLASS_IDENTITY_PICKLE_PROBE.json
```

Result with Python `3.10.12` on CPU:

- An ordinary configuration instance serializes and transfers as pickle bytes
  through a pipe to a fresh isolated interpreter. The worker restores
  `hidden_size = 4096` and uses its own imported `DemoConfig` class.
- A repeated ordinary import preserves class identity. Reloading the unchanged
  source file creates a new class object with the same module and class names.
- The old instance keeps the old class, so pickling it raises `PicklingError`
  in the sender before returning a complete pickle.
- A newly created instance of the currently published class transfers normally.

The example imports only locally generated code and the Python standard
library. It does not run a Hugging Face or MAX model. The output is
[`CLASS_IDENTITY_PICKLE_PROBE.json`](CLASS_IDENTITY_PICKLE_PROBE.json); source
versions and input/output hashes are in the evidence record's
`class_identity_followup` entry. The diagram buttons switch illustrations of
these outcomes and do not execute Python in the browser.
