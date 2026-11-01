"""Generate a separate, offline text-tree reader with exact source links."""
from __future__ import annotations

import html
import json
import re

from call_stack_refs import FILES, REFS

COMMANDS = """# Explicit task: the task-inference branch is skipped.
max serve --model meta-llama/Llama-2-7b-hf --task text_generation

# Infer task: taken when the effective task is UNDEFINED (no config-file task).
max serve --model meta-llama/Llama-2-7b-hf"""


def startup_html(src):
    return ('<details class="startup-text-map" id="startup-map"><summary>'
            '<strong>Startup map</strong><span> · API process / Hub cache / model worker · SOURCE</span>'
            '</summary><pre tabindex="0">' + html.escape((src / "startup_map.txt").read_text()) + '</pre></details>')


def build_call_stack(data, task, src, span_finder):
    refs = {}
    for key, spec in REFS.items():
        file = FILES[spec["alias"]]
        source = data["files"][file]["text"]
        lines = source.splitlines()
        symbol = spec.get("symbol")
        start, end = span_finder(source, symbol) if symbol else (spec["line"], spec["line"])
        line = start
        if needle := spec.get("needle"):
            body = "\n".join(lines[start - 1:end])
            if body.count(needle) != 1:
                raise ValueError(f"Call-stack target must match once: {key}: {symbol}: {needle!r}")
            line += body[:body.index(needle)].count("\n")
        refs[key] = {**spec, "file": file, "line": line, "start": start, "end": end, "text": lines[line - 1]}

    def label(ref):
        return f'{ref["label"]} [{ref["alias"]}:L{ref["line"]}]'

    def link(key):
        ref = refs[key]
        return (f'<a href="sources/{ref["file"]}#L{ref["line"]}" data-ref="{key}" '
                f'title="{html.escape(ref["kind"] + ": " + ref["file"])}">'
                + html.escape(label(ref)) + '</a>')

    token = re.compile(r"\{\{([a-z_][a-z_0-9]*)\}\}")
    sections = []
    used = set()
    raw = (src / "call_stack_tree.txt").read_text()
    pieces = re.split(r"^@@ (1\.1\.[1-6]) \| (.+)\n", raw, flags=re.M)
    if pieces[0]:
        raise ValueError("Call stack must begin with a numbered section")
    for offset in range(1, len(pieces), 3):
        number, title, content = pieces[offset:offset + 3]
        content = content.strip("\n") + "\n"
        for key in token.findall(content):
            if key not in refs:
                raise ValueError("Unknown source reference: " + key)
            used.add(key)
        plain = token.sub(lambda m: label(refs[m[1]]), content)
        rows = []
        for line in content.splitlines():
            rendered = token.sub(lambda m: link(m[1]), html.escape(line))
            rows.append('<span class="tree-line">' + rendered + '</span>')
        id = "stack-" + number.replace(".", "-")
        sections.append(dict(number=number, title=title, id=id, text=plain,
            html=f'<section class="stack-section" id="{id}"><h2>QnA {number} · {html.escape(title)}</h2>'
                 '<pre class="text-tree" tabindex="0"><code>' + "\n".join(rows) + '\n</code></pre></section>'))
    if [s["number"] for s in sections] != [f"1.1.{n}" for n in range(1, 7)]:
        raise ValueError("Expected all six call-stack sections in order")
    if set(refs) != used:
        raise ValueError("Unused call-stack references: " + str(set(refs) - used))
    intro = (f'LLAMA 2 7B / MAX SERVE CALL STACKS\nSOURCE revision: {data["revision"]}\n'
             'Selected source-derived call paths; illustrative values, not a captured execution trace.\n'
             'Scope: standard API/worker route, single device, built-in Graph API.\n\n' + COMMANDS + '\n\n')
    text = intro + (src / "startup_map.txt").read_text() + "\n"
    markdown = "# Llama 2 7B / MAX serve call stacks\n\n[Interactive text-tree window](call_stack.html).\n\n"
    markdown += 'SOURCE revision: ' + data["revision"] + '\n\n'
    for section in sections:
        text += f'QnA {section["number"]} | {section["title"]}\n' + section["text"] + "\n"
        markdown += f'## QnA {section["number"]} · {section["title"]}\n\n'
        markdown += chr(96) * 3 + 'text\n' + section["text"] + chr(96) * 3 + '\n\n'
    map_rows = []
    text += "SOURCE MAP (paths relative to the repository root)\n"
    markdown += "## Source map\n\n"
    for alias, file in FILES.items():
        text += alias + " = " + file + "\n"
        markdown += f'- {alias}: [{file}](sources/{file})\n'
        map_rows.append(f'<tr><td>{alias}</td><td><a href="sources/{file}" data-file="{file}">{file}</a></td></tr>')
    (task / "CALL_STACKS.txt").write_text(text)
    (task / "CALL_STACKS.md").write_text(markdown)
    manifest = dict(revision=data["revision"], references=refs,
                    sections=[{k:v for k,v in s.items() if k != "html"} for s in sections])
    (task / "call_stack_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    viewer = dict(**manifest, files={file:data["files"][file] for file in FILES.values()}, text=text)
    payload = json.dumps(viewer, ensure_ascii=False).replace("<", "\\u003c")
    options = ''.join(f'<option value="{file}">{file}</option>' for file in viewer["files"])
    nav = ''.join(f'<a href="#{s["id"]}">QnA {s["number"]}</a>' for s in sections)
    css = (src / "style.css").read_text() + "\n" + (src / "call_stack.css").read_text()
    js = (src / "call_stack.js").read_text()
    output = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Llama 2 7B · Call stack tree</title><style>{css}</style></head>
<body class="stack-page"><a class="skip-link" href="#stack-main">Skip to call trees</a>
<header class="stack-header"><div><span class="badge">SOURCE</span><strong>Call stack tree · QnA 1.1</strong></div>
<a href="index.html#qna-1-1">Main guide ↗</a></header>
<main class="stack-main" id="stack-main">
<h1>Llama 2 7B: who calls whom?</h1>
<p class="stack-intro">Six text trees connect CLI arguments, registry objects, Hugging Face assets, server startup and model compilation.
Function links open complete original source; links labeled <b>call site</b> highlight the invoking line.</p>
<p class="stack-evidence">SOURCE · {data["revision"]} · Standard API/worker path, one device, Graph API.
Selected calls and illustrative values; no model execution was recorded. External APIs link to their MAX invocation.</p>
<pre class="stack-commands"><code>{html.escape(COMMANDS)}</code></pre>
<p class="stack-key"><code>+--</code> call/branch · <code>-&gt;</code> condensed call chain or labeled data flow ·
<code>=&gt;</code> process handoff · <code>[ALIAS:L123]</code> exact source location.
API and worker entries have separate stacks. The trees repeat shared calls to keep each QnA readable.</p>
{startup_html(src)}
<details class="stack-source-map"><summary><strong>Source map</strong> · file aliases used in the trees</summary>
<table><thead><tr><th>Alias</th><th>Complete original file</th></tr></thead><tbody>{''.join(map_rows)}</tbody></table></details>
<nav class="stack-nav" aria-label="Call-stack sections">{nav}</nav>
<div class="stack-tools"><label>Find in trees <input id="tree-search" type="search" placeholder="Function, object or component"></label>
<button id="tree-next" type="button">Next match</button><output id="tree-count" aria-live="polite"></output>
<button id="copy-tree" type="button">Copy all text</button><a href="CALL_STACKS.txt" download>Download .txt</a></div>
<p class="stack-scroll-hint">Scroll inside a tree to read long lines. Indentation and source code are preserved.</p>
{''.join(s["html"] for s in sections)}
</main>
<dialog id="source-dialog" aria-labelledby="source-dialog-title"><div class="dialog-head"><div><span class="eyebrow">COMPLETE ORIGINAL FILE</span>
<h2 id="source-dialog-title">Source explorer</h2></div><button id="close-explorer" aria-label="Close source explorer">Close ✕</button></div>
<label class="file-select-label">Source file<select id="file-select">{options}</select></label>
<div class="file-tools"><label>Find in file <input id="file-search" type="search" placeholder="Symbol or text"></label>
<button id="find-next">Next match</button><span id="find-count" aria-live="polite"></span><a id="download-full" download>Download original file</a></div>
<div id="file-provenance"></div><pre id="full-file" class="source-code" tabindex="0"><code></code></pre></dialog>
<div id="toast" role="status" aria-live="polite"></div>
<script id="stack-data" type="application/json">{payload}</script><script>{js}</script></body></html>'''
    (task / "call_stack.html").write_text(output)
    return len(refs)
