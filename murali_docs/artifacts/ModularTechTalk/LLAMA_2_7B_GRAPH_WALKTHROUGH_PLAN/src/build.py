#!/usr/bin/env python3
"""Build an offline guide and Markdown companion from verified repository source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

from cells import CELLS, REVISION

TASK = Path(__file__).resolve().parent.parent
REPO = TASK.parents[3]
SRC = TASK / "src"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def python_span(text: str, symbol: str) -> tuple[int, int]:
    """Find a complete Python definition or assignment, including decorators."""
    node = ast.parse(text)
    for part in symbol.split("."):
        matches = []
        for child in node.body:
            if getattr(child, "name", None) == part:
                matches.append(child)
            elif isinstance(child, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == part for t in child.targets):
                    matches.append(child)
        if len(matches) != 1:
            raise ValueError(f"Expected one Python definition for {symbol}: {part}")
        node = matches[0]
    starts = [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]
    return min(starts), node.end_lineno


def mask_cpp(text: str) -> str:
    """Mask comments and quoted literals while preserving offsets and line breaks."""
    out = list(text)
    i = 0
    while i < len(text):
        end = i
        if text.startswith("//", i):
            end = text.find("\n", i)
            if end == -1:
                end = len(text)
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end == -1:
                raise ValueError("Unterminated C++ comment")
            end += 2
        elif (raw := re.match(r'R"([^ ()\\\t\r\n]{0,16})\(', text[i:])):
            marker = ")" + raw.group(1) + '"'
            end = text.find(marker, i + raw.end())
            if end == -1:
                raise ValueError("Unterminated C++ raw string")
            end += len(marker)
        elif text[i] in "\"'":
            quote = text[i]
            end = i + 1
            while end < len(text):
                if text[end] == "\\":
                    end += 2
                elif text[end] == quote:
                    end += 1
                    break
                else:
                    end += 1
        if end > i:
            for j in range(i, min(end, len(text))):
                if out[j] != "\n":
                    out[j] = " "
            i = end
        else:
            i += 1
    return "".join(out)


def cpp_span(text: str, needle: str, prefix: int) -> tuple[int, int]:
    """Find a full C++ definition by balanced braces, skipping declarations."""
    masked = mask_cpp(text)
    candidates = []
    for match in re.finditer(re.escape(needle), text):
        brace = masked.find("{", match.start())
        semicolon = masked.find(";", match.start())
        if brace == -1 or (semicolon != -1 and semicolon < brace):
            continue
        depth = 0
        for end in range(brace, len(masked)):
            depth += (masked[end] == "{") - (masked[end] == "}")
            if depth == 0:
                start_line = text.count("\n", 0, match.start()) + 1 - prefix
                end_line = text.count("\n", 0, end) + 1
                candidates.append((start_line, end_line))
                break
    if len(candidates) != 1:
        raise ValueError(f"Expected one C++ definition for {needle}: {candidates}")
    return candidates[0]


def extract_sources() -> dict:
    from call_stack_refs import FILES
    files = {}
    cells = []
    extra_files = [
        "max/python/max/_entrypoints/pipelines.py",
        "max/python/max/_entrypoints/cli/config.py",
        "max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py",
        "max/python/max/serve/api_server.py",
        "max/python/max/serve/config.py",
        "max/python/max/serve/pipelines/model_worker.py",
        "max/python/max/graph/weights/load.py",
        "max/python/max/pipelines/lib/pipeline_args.py",
        "max/python/max/pipelines/lib/config/config.py",
        "max/python/max/pipelines/lib/interfaces/pipeline_model.py",
        "max/python/max/pipelines/lib/memory_estimation.py",
        "max/python/max/pipelines/lib/pipeline_variants/text_generation.py",
        "max/python/max/pipelines/__init__.py",
        "max/python/max/pipelines/architectures/__init__.py",
        "max/python/max/pipelines/lib/arch_lookup.py",
        "max/python/max/pipelines/lib/registry.py",
        "max/python/max/pipelines/lib/_hf_config.py",
        "max/python/max/pipelines/lib/config/model_config.py",
        "max/python/max/pipelines/weights/hf_utils.py",
        "max/python/max/pipelines/lib/tokenizer.py",
    ]
    for file in [c["file"] for c in CELLS] + extra_files + list(FILES.values()):
        if file not in files:
            raw = (REPO / file).read_bytes()
            pinned = subprocess.check_output(["git", "show", f"{REVISION}:{file}"], cwd=REPO)
            if raw != pinned:
                raise ValueError(f"Working source differs from recorded revision: {file}")
            target = TASK / "sources" / file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            files[file] = dict(path=file, snapshot=f"sources/{file}",
                               sha256=digest(raw), text=raw.decode("utf-8"))
    for spec in CELLS:
        file = spec["file"]
        text = files[file]["text"]
        start, end = (python_span(text, spec["symbol"]) if spec["language"] == "python"
                      else cpp_span(text, spec["needle"], spec.get("prefix", 0)))
        code = "".join(text.splitlines(keepends=True)[start - 1:end])
        annotations = []
        for item in spec["notes"]:
            if item["needle"] not in code:
                raise ValueError(f"Missing annotation in {spec['id']}: {item['needle']}")
            line = start + code[:code.index(item["needle"])].count("\n")
            annotations.append(dict(title=item["title"], text=item["text"],
                                    start=line, end=min(end, line + item["span"] - 1)))
        cells.append(dict(id=spec["id"], file=file, symbol=spec["symbol"],
                          purpose=spec["purpose"], language=spec["language"],
                          evidence=spec["evidence"], start=start, end=end,
                          code=code, sha256=digest(code.encode()), notes=annotations))
    manifest = dict(revision=REVISION, files=[{k: v for k, v in f.items() if k != "text"}
                                              for f in files.values()],
                    cells=[{k: v for k, v in c.items() if k != "code"} for c in cells])
    (TASK / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(revision=REVISION, files=files, cells=cells)


def code_html(cell: dict) -> str:
    esc = html.escape
    rows = "\n".join(
        f'<span class="code-line" data-line="{line}"><span class="line-number" aria-hidden="true">{line}</span>'
        f'<span class="line-text">{esc(text)}</span></span>'
        for line, text in enumerate(cell["code"].splitlines(), cell["start"])
    )
    notes = "".join(
        f'<button class="annotation" data-cell="{cell["id"]}" data-start="{n["start"]}" data-end="{n["end"]}">'
        f'<strong>{esc(n["title"])}</strong><span>{esc(n["text"])}</span>'
        f'<small>Lines {n["start"]}–{n["end"]}</small></button>' for n in cell["notes"]
    )
    return f'''<details class="code-cell" id="code-{cell['id']}" data-cell="{cell['id']}">
<summary><span class="disclosure" aria-hidden="true"></span><span class="cell-label"><strong>{esc(cell['symbol'])}</strong><span>{esc(cell['purpose'])}</span></span><span class="badge">{cell['evidence']}</span><span class="line-count">{cell['end'] - cell['start'] + 1} lines</span></summary>
<div class="cell-meta"><code>{esc(cell['file'])}</code><span>L{cell['start']}–L{cell['end']} · {REVISION[:12]}</span></div>
<div class="cell-tools"><button data-copy="{cell['id']}">Copy complete source</button><button data-file="{esc(cell['file'])}" data-file-line="{cell['start']}">View whole file</button><a href="sources/{esc(cell['file'])}" download>Download file</a><a href="#{'code-' + cell['id']}" class="permalink">Link to cell</a></div>
<div class="cell-body"><pre class="source-code" aria-label="Complete original {esc(cell['symbol'])}"><code>{rows}</code></pre><aside class="annotations" aria-label="Code explanations">{notes}<p class="exact-note">Complete source. Annotations are separate from the original code.</p></aside></div>
</details>'''


def code_markdown(cell: dict) -> str:
    lines = [f'<a id="code-{cell["id"]}"></a>', "",
             f'<details><summary><strong>{cell["symbol"]}</strong> — {cell["end"] - cell["start"] + 1} complete lines · {cell["evidence"]}</summary>', "",
             cell["purpose"], "",
             f'Original: [{cell["file"]}](../../../../{cell["file"]}#L{cell["start"]}); lines {cell["start"]}–{cell["end"]}.',
             f'Revision: `{REVISION}`. [Complete file](sources/{cell["file"]}).', "",
             f'```{cell["language"]}', cell["code"].rstrip("\n"), "```", "",
             "| Source lines | Explanation |", "|---|---|"]
    for n in cell["notes"]:
        lines.append(f'| {n["start"]}–{n["end"]} | **{n["title"]}** — {n["text"]} |')
    return "\n".join(lines + ["", "</details>", ""])


def build_guide(data: dict) -> None:
    import markdown
    from call_stack import READER_NAME, build_call_stack, startup_html
    from widgets import WIDGETS, stage_data

    reference_count = build_call_stack(data, TASK, SRC, python_span)

    cells = {c["id"]: c for c in data["cells"]}
    template = (SRC / "content.md").read_text()
    qna = (SRC / "qna.md").read_text()
    autoconfig_qna = (SRC / "qna_autoconfig.md").read_text()
    qna_summary = '<strong>QnA 1</strong><span>Package name · Hugging Face checkpoint · Architecture registration</span>'
    nested_cells = [
        dict(id="qna-1-1-1", number="1.1.1", title="HF config loading",
             question="Who calls load_huggingface_config and AutoConfig?",
             file="qna_1_1_1_config.md"),
        dict(id="qna-1-1-2", number="1.1.2", title="Architecture registration",
             question="When does MAX register and import the Llama implementation?",
             file="qna_1_1_2_registry.md"),
        dict(id="qna-1-1-3", number="1.1.3", title="Tokenizer loading",
             question="How does max serve reach AutoTokenizer?",
             file="qna_1_1_3_tokenizer.md"),
        dict(id="qna-1-1-4", number="1.1.4", title="Generation defaults",
             question="How does GenerationConfig reach MAX sampling settings?",
             file="qna_1_1_4_generation.md"),
        dict(id="qna-1-1-5", number="1.1.5", title="Weight files",
             question="When are weight paths resolved, downloaded, and loaded?",
             file="qna_1_1_5_weights.md"),
        dict(id="qna-1-1-6", number="1.1.6", title="Llama3Config lifecycle",
             question="When are initialize and finalize called?",
             file="qna_1_1_6_llama_config.md"),
    ]
    child_content = {}
    for item in nested_cells:
        placeholder = f'{{{{nested:{item["id"]}}}}}'
        if autoconfig_qna.count(placeholder) != 1:
            raise ValueError(f"Expected one placeholder for {item['number']}")
        child_content[item["id"]] = (SRC / item["file"]).read_text()

    def nested_cell_html(item: dict) -> str:
        title = html.escape(f'QnA {item["number"]} · {item["title"]}')
        question = html.escape(item["question"])
        content = markdown.markdown(
            child_content[item["id"]],
            extensions=["tables", "fenced_code", "attr_list"],
        )
        return (
            f'<details class="nested-qna" id="{item["id"]}">'
            '<summary><span class="disclosure" aria-hidden="true"></span>'
            '<span class="nested-qna-label">'
            f'<strong>{title}</strong><span>{question}</span>'
            '</span></summary><div class="nested-qna-body">'
            f'{content}</div></details>'
        )

    def nested_cell_markdown(item: dict) -> str:
        title = html.escape(f'QnA {item["number"]} · {item["title"]}')
        question = html.escape(item["question"])
        return (
            f'<details id="{item["id"]}"><summary><strong>{title}</strong> '
            f'— {question}</summary>\n\n'
            f'{child_content[item["id"]]}\n</details>'
        )

    qna_1_1_html_body = markdown.markdown(
        autoconfig_qna, extensions=["tables", "fenced_code", "attr_list"]
    )
    qna_1_1_markdown_body = autoconfig_qna
    qna_1_1_html_body = qna_1_1_html_body.replace("<p>{{startup-map}}</p>", startup_html(SRC))
    qna_1_1_markdown_body = qna_1_1_markdown_body.replace(
        "{{startup-map}}", '<details><summary>Startup map · SOURCE</summary>\n\n'
        + chr(96) * 3 + "text\n" + (SRC / "startup_map.txt").read_text()
        + chr(96) * 3 + "\n\n</details>")
    for item in nested_cells:
        placeholder = f'{{{{nested:{item["id"]}}}}}'
        html_slot = f"<p>{placeholder}</p>"
        if qna_1_1_html_body.count(html_slot) != 1:
            raise ValueError(f"Expected one rendered placeholder for {item['number']}")
        qna_1_1_html_body = qna_1_1_html_body.replace(
            html_slot, nested_cell_html(item)
        )
        qna_1_1_markdown_body = qna_1_1_markdown_body.replace(
            placeholder, nested_cell_markdown(item)
        )

    nested_summary = (
        '<strong>QnA 1.1 · Checkpoint-to-model call paths</strong>'
        '<span>Config · registration · tokenizer · generation defaults · weights · Llama3Config</span>'
    )
    nested_question = "How does max serve prepare a Llama checkpoint and model?"
    nested_qna_html = (
        '<details class="nested-qna" id="qna-1-1">'
        f'<summary><span class="disclosure" aria-hidden="true"></span><span class="nested-qna-label">{nested_summary}</span>'
        f'<a class="stack-launch" id="open-call-stack" href="{READER_NAME}" target="_blank" rel="noopener" '
        'aria-label="Open the combined MAX Serve startup call stack in a new window">Call stack tree ↗</a></summary>'
        '<div class="nested-qna-body">'
        + qna_1_1_html_body
        + '</div></details>'
    )
    nested_qna_markdown = (
        '<details id="qna-1-1"><summary><strong>QnA 1.1 · '
        'Checkpoint-to-model call paths</strong> — '
        f'{nested_question}</summary>\n\n'
        + qna_1_1_markdown_body
        + '\n</details>'
    )
    if qna.count("{{nested:qna-1-1}}") != 1:
        raise ValueError("QnA 1.1 must appear once inside QnA 1")
    qna_html_content = qna.replace("{{nested:qna-1-1}}", nested_qna_html)
    qna_markdown_content = qna.replace("{{nested:qna-1-1}}", nested_qna_markdown)
    qna_html = (
        '<details class="qna-cell" id="qna-1">'
        '<summary><span class="disclosure" aria-hidden="true"></span>'
        f'<span class="cell-label">{qna_summary}</span>'
        '<span class="badge">SOURCE</span><span class="line-count">3 questions</span></summary>'
        '<div class="qna-body">'
        + markdown.markdown(qna_html_content, extensions=["tables", "fenced_code", "attr_list"])
        + '</div></details>'
    )
    qna_markdown = (
        '<details id="qna-1"><summary><strong>QnA 1</strong> — Package name, '
        'Hugging Face checkpoint, and architecture registration</summary>\n\n'
        + qna_markdown_content + '\n</details>\n'
    )
    used = re.findall(r"\{\{cell:([a-z_]+)\}\}", template)
    if len(used) != len(set(used)) or set(used) != set(cells):
        raise ValueError(f"Every source cell must occur once: missing {set(cells)-set(used)}, unknown {set(used)-set(cells)}")

    def materialize(kind: str) -> str:
        body = re.sub(r"\{\{cell:([a-z_]+)\}\}",
                      lambda m: code_html(cells[m[1]]) if kind == "html" else code_markdown(cells[m[1]]), template)
        body = body.replace("{{qna:1}}", qna_html if kind == "html" else qna_markdown)
        return re.sub(r"\{\{widget:([a-z_]+)\}\}", lambda m: WIDGETS[m[1]][kind], body)

    intro = f"""# Llama 2 7B: from graph to executable\n\nEvidence: `SOURCE` unless marked otherwise. Source revision: `{REVISION}`.\n\nRead [the interactive guide](index.html) for selectable diagrams and code-line highlighting.\nThis companion contains the same six chapters and complete expandable source functions.\n\nCode is from the pinned checkout. State/buffer examples are illustrations; repository tests are inspected, not run.\nNo Llama inference or compiler dump is presented as a measured result.\n\n"""
    (TASK / "WALKTHROUGH.md").write_text(intro + materialize("markdown") + "\n")
    body = materialize("html")
    parts = re.split(r"(?=^## [1-6]\. )", body, flags=re.M)
    chapters = []
    for part in parts:
        if not part.strip():
            continue
        match = re.match(r"## ([1-6])\. (.+)", part)
        if not match:
            raise ValueError("Chapter must start with a numbered heading")
        number, title = match.groups()
        rendered = markdown.markdown(part, extensions=["tables", "fenced_code", "attr_list"])
        def source_link(match):
            file, line = match[1], match[2] or "1"
            if file not in data["files"]:
                raise ValueError(f"Guide source link lacks a complete snapshot: {file}")
            return (f'href="sources/{file}#L{line}" data-file="{file}" '
                    f'data-file-line="{line}"')
        rendered = re.sub(r'href="\.\./\.\./\.\./\.\./([^"#]+)(?:#L(\d+))?"', source_link, rendered)
        chapters.append((number, title, f'<section id="chapter-{number}" class="chapter" data-chapter="{number}" aria-label="{html.escape(title)}">{rendered}</section>'))
    nav = "".join(f'<a href="#chapter-{n}" data-nav="{n}"><span>{n.zfill(2)}</span>{html.escape(t)}</a>' for n, t, _ in chapters)
    files = list(data["files"].values())
    options = "".join(f'<option value="{html.escape(f["path"])}">{html.escape(f["path"])}</option>' for f in files)
    data["stages"] = stage_data()
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    css = (SRC / "style.css").read_text() + "\n" + (SRC / "call_stack.css").read_text()
    js = (SRC / "app.js").read_text()
    output = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="Llama 2 7B in MAX: graph construction, finalization, native compilation, and execution, with complete expandable source."><title>Llama 2 7B · Inside the MAX graph compiler</title><style>{css}</style></head>
<body><a class="skip-link" href="#main">Skip to the walkthrough</a>
<header class="topbar"><div class="topbar-start"><button id="toggle-sidebar" class="sidebar-toggle" type="button" aria-controls="chapter-sidebar" aria-expanded="true" title="Hide sidebar" hidden><svg class="sidebar-icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="2" y="3" width="20" height="18" rx="2"/><path d="M8 3v18"/><path class="sidebar-chevron" d="m16 8-4 4 4 4"/></svg><span>Hide sidebar</span></button><a href="#top" class="brand" aria-label="MAX source study: back to top"><span class="brand-mark">M</span><span class="brand-title">MAX / SOURCE STUDY</span></a></div><div class="top-actions"><a href="WALKTHROUGH.md">Markdown</a><a href="EXPLANATION_RESEARCH.md">Research</a><button id="open-explorer">Source explorer</button><button id="print-guide" title="Print the guide">Print</button></div></header>
<div class="page-shell"><aside class="sidebar" id="chapter-sidebar"><div class="sidebar-caption">THE READING PATH</div><nav aria-label="Chapters">{nav}</nav><div class="sidebar-note"><span class="badge">SOURCE</span><p>One model. Six stages.<br>Complete original code.</p><code>{REVISION[:12]}</code></div><div class="sidebar-actions"><button id="expand-chapter">Expand chapter code</button><button id="collapse-code">Collapse all code</button></div><a class="source-map-link" href="SOURCE_MAP.md">Compact source map ↗</a></aside>
<main id="main"><div id="top" class="hero"><div class="eyebrow">LLAMA 2 7B · SINGLE DEVICE · GRAPH API</div><h1>From a Llama model<br>to an executable graph.</h1><p class="lead">See what MAX records, what finalization finishes, and how the native compiler turns that graph into a model you can execute.</p><div class="hero-chips"><span>6 chapters</span><span>{len(cells)} complete source cells</span><span>{len(files)} full files</span><span>Works offline</span></div><div class="hero-contract"><div><small>FINALIZE</small><strong>Finish the IR contract</strong></div><div><small>COMPILE</small><strong>Produce an artifact</strong></div><div><small>INITIALIZE</small><strong>Bind executable state</strong></div></div><p class="evidence-note">Source-derived explanations at revision {REVISION[:12]}. Diagrams illustrate the code; they are not captured Llama execution or measured GPU timings.</p></div>
<noscript><div class="callout">The chapters and complete code cells work without JavaScript. Use the Markdown companion for static diagram fallbacks. JavaScript enables line highlights, source search, and the diagram controls.</div></noscript>
{''.join(c[2] for c in chapters)}
<footer class="guide-footer"><strong>Continue in the repository.</strong><p><a href="SOURCE_MAP.md">Source map</a> · <a href="EXPLANATION_RESEARCH.md">Teaching references</a> · <a href="DESIGN_PLAN.md">Design plan</a> · <a href="source_manifest.json">Source manifest</a></p><p>All source cells were extracted from the recorded revision; comments and docstrings are retained. The full-file viewer includes the original license headers.</p></footer></main></div>
<dialog id="source-dialog" aria-labelledby="source-dialog-title"><div class="dialog-head"><div><span class="eyebrow">COMPLETE ORIGINAL FILE</span><h2 id="source-dialog-title">Source explorer</h2></div><button id="close-explorer" aria-label="Close source explorer">Close ✕</button></div><label class="file-select-label">Source file<select id="file-select">{options}</select></label><div class="file-tools"><label>Find in file <input id="file-search" type="search" placeholder="Symbol or text" autocomplete="off"></label><button id="find-next">Next match</button><span id="find-count" aria-live="polite"></span><a id="download-full" download>Download original file</a></div><div id="file-provenance"></div><pre id="full-file" class="source-code" tabindex="0"><code></code></pre></dialog>
<div id="toast" role="status" aria-live="polite"></div><script id="guide-data" type="application/json">{payload}</script><script>{js}</script></body></html>'''
    (TASK / "index.html").write_text(output)
    print(f"Built six chapters, {len(cells)} complete cells, and {len(files)} full-file snapshots.")
    print(f"Built one combined MAX Serve startup call tree: {reference_count} verified source targets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    data = extract_sources()
    print(f"Verified {len(data['cells'])} source selections at {REVISION[:12]}.")
    if not args.extract_only:
        build_guide(data)
