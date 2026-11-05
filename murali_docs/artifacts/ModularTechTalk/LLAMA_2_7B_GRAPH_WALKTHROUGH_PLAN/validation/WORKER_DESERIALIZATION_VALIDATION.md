# Worker deserialization explanation validation

- Date: 2026-11-05.
- Evidence: `SOURCE` for MAX behavior; `CPU-RUN` for browser checks.
- Saved MAX source revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
- Documentation baseline: `36b70c3f4f5e96314076532baf80f5c6cfadca79`.
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
outside the filesystem sandbox. The final run passed all nine check groups:

- Exact link placement, one occurrence and opening in a new tab.
- Visible tree, copy payload, text/Markdown exports and manifest agree.
- Complete original answer, code excerpt, diagrams and reconstruction table.
- Added derived-state explanation and keyboard disclosure controls.
- Existing local navigation and saved-source links resolve.
- Layout at 1440px, 390px and 320px; horizontal scrolling stays inside diagrams.
- Print colours and keyboard navigation.
- Existing source explorer still highlights `__setstate__()` at line 1024.
- Direct `file://` use and return navigation with JavaScript disabled.

The final contrast check examined 158 samples per screen width, with a minimum
text ratio of 6.621:1. Print checked 156 samples, with a minimum of 6.233:1.
No contrast failures, JavaScript errors or external resource requests occurred.
Temporary screenshots and the detailed report are in
`/tmp/max-worker-deserialization-checks/`.
