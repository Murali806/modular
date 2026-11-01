"""Generate a separate, offline text-tree reader with exact source links."""
from __future__ import annotations

import html
import json
import re

from call_stack_refs import FILES, REFS

READER_NAME = "max_serve_llama2_7b_startup_call_stack.html"
TEXT_NAME = "MAX_SERVE_LLAMA2_7B_STARTUP_CALL_STACK.txt"
MARKDOWN_NAME = "MAX_SERVE_LLAMA2_7B_STARTUP_CALL_STACK.md"

COMMANDS = """# Explicit task: the task-inference branch is skipped.
max serve --model meta-llama/Llama-2-7b-hf --task text_generation

# Infer task: taken when the effective task is UNDEFINED (no config-file task).
max serve --model meta-llama/Llama-2-7b-hf"""

LOCAL_LINKS = {
    "manifest_roles": {
        "href": "link_htmls/model_manifest_roles_and_diffusers.html",
        "label": "Model manifest roles, components, and Diffusers ↗",
        "text": (
            "Model manifest roles, components, and Diffusers ↗ "
            "(link_htmls/model_manifest_roles_and_diffusers.html)"
        ),
        "title": "Open the explanation of main and component manifests",
    },
    "repo_examples": {
        "href": "link_htmls/huggingface_repo_handles_and_external_weights.html",
        "label": "Hugging Face repo handles and external weights ↗",
        "text": (
            "Hugging Face repo handles and external weights ↗ "
            "(link_htmls/huggingface_repo_handles_and_external_weights.html)"
        ),
        "title": "Open examples of Hub repo URLs and separate weight repositories",
    },
}


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
    local_token = re.compile(r"\[\[doc:([a-z_][a-z_0-9]*)\]\]")

    def local_link(key):
        spec = LOCAL_LINKS[key]
        return (
            f'<a class="tree-doc-link" href="{html.escape(spec["href"], quote=True)}" '
            f'target="_blank" rel="noopener noreferrer" '
            f'title="{html.escape(spec["title"], quote=True)}">'
            f'{html.escape(spec["label"])}</a>'
        )

    sections = []
    used = set()
    used_local_links = set()
    raw = (src / "call_stack_tree.txt").read_text()
    pieces = re.split(r"^@@ (1\.1) \| (.+)\n", raw, flags=re.M)
    if pieces[0]:
        raise ValueError("Call stack must begin with a numbered section")
    for offset in range(1, len(pieces), 3):
        number, title, content = pieces[offset:offset + 3]
        content = content.strip("\n") + "\n"
        for key in token.findall(content):
            if key not in refs:
                raise ValueError("Unknown source reference: " + key)
            used.add(key)
        for key in local_token.findall(content):
            if key not in LOCAL_LINKS:
                raise ValueError("Unknown local document link: " + key)
            used_local_links.add(key)
        plain = token.sub(lambda m: label(refs[m[1]]), content)
        plain = local_token.sub(lambda m: LOCAL_LINKS[m[1]]["text"], plain)
        rows = []
        for line in content.splitlines():
            rendered = token.sub(lambda m: link(m[1]), html.escape(line))
            rendered = local_token.sub(lambda m: local_link(m[1]), rendered)
            rows.append('<span class="tree-line">' + rendered + '</span>')
        id = "stack-startup"
        sections.append(dict(number=number, title=title, id=id, text=plain,
            html=f'<section class="stack-section" id="{id}"><h2>{html.escape(title)}</h2>'
                 '<pre class="text-tree" tabindex="0"><code>' + "\n".join(rows) + '\n</code></pre></section>'))
    if len(sections) != 1 or sections[0]["number"] != "1.1":
        raise ValueError("Expected one combined startup call stack covering QnA 1.1.1-1.1.6")
    if set(refs) != used:
        raise ValueError("Unused call-stack references: " + str(set(refs) - used))
    if set(LOCAL_LINKS) != used_local_links:
        raise ValueError("Unused local document links: " + str(set(LOCAL_LINKS) - used_local_links))
    intro = (f'MAX SERVE / LLAMA 2 7B STARTUP CALL STACK\nSOURCE revision: {data["revision"]}\n'
             'Selected source-derived call paths; illustrative values, not a captured execution trace.\n'
             'Purpose: CLI startup -> checkpoint/config -> model loading -> graph compile/init -> HTTP readiness.\n'
             'Scope: QnA 1.1.1-1.1.6 combined; standard API/worker route, single device, built-in Graph API.\n\n' + COMMANDS + '\n\n')
    text = intro
    markdown = f"# MAX Serve / Llama 2 7B startup call stack\n\n[Interactive call-stack window]({READER_NAME}).\n\n"
    markdown += 'SOURCE revision: ' + data["revision"] + '\n\n'
    markdown += chr(96) * 3 + 'bash\n' + COMMANDS + '\n' + chr(96) * 3 + '\n\n'
    for section in sections:
        text += 'COMBINED STARTUP CALL STACK (QnA 1.1.1-1.1.6)\n' + section["text"] + "\n"
        markdown += '## Combined startup call stack (QnA 1.1.1-1.1.6)\n\n'
        markdown += chr(96) * 3 + 'text\n' + section["text"] + chr(96) * 3 + '\n\n'
    map_rows = []
    text += "SOURCE MAP (paths relative to the repository root)\n"
    markdown += "## Source map\n\n"
    for alias, file in FILES.items():
        text += alias + " = " + file + "\n"
        markdown += f'- {alias}: [{file}](sources/{file})\n'
        map_rows.append(f'<tr><td>{alias}</td><td><a href="sources/{file}" data-file="{file}">{file}</a></td></tr>')
    (task / TEXT_NAME).write_text(text)
    (task / MARKDOWN_NAME).write_text(markdown)
    manifest = dict(revision=data["revision"], references=refs,
                    sections=[{k:v for k,v in s.items() if k != "html"} for s in sections])
    (task / "call_stack_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    viewer = dict(**manifest, files={file:data["files"][file] for file in FILES.values()}, text=text)
    payload = json.dumps(viewer, ensure_ascii=False).replace("<", "\\u003c")
    options = ''.join(f'<option value="{file}">{file}</option>' for file in viewer["files"])
    css = (src / "style.css").read_text() + "\n" + (src / "call_stack.css").read_text()
    js = (src / "call_stack.js").read_text()
    output = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MAX Serve · Llama 2 7B startup call stack</title><style>{css}</style></head>
<body class="stack-page"><a class="skip-link" href="#stack-startup">Skip to the combined call tree</a>
<header class="stack-header"><div><span class="badge">SOURCE</span><strong>MAX Serve · Llama 2 7B startup call stack</strong></div>
<a href="index.html#qna-1-1">Main guide ↗</a></header>
<main class="stack-main" id="stack-main">
<h1>From the serve command to a ready model.</h1>
<p class="stack-intro">One continuous call tree combines QnA 1.1.1–1.1.6: CLI parsing, checkpoint configuration,
architecture registration, tokenizer/defaults, model weights, graph compilation and HTTP readiness.
Function links open complete original source; links labeled <b>call site</b> highlight the invoking line.</p>
<p class="stack-evidence">SOURCE · {data["revision"]} · Standard API/worker path, one device, Graph API.
Selected calls and illustrative values; no model execution was recorded. External APIs link to their MAX invocation.</p>
<pre class="stack-commands"><code>{html.escape(COMMANDS)}</code></pre>
<p class="stack-key"><code>+--</code> call/branch · <code>-&gt;</code> condensed call chain or labeled data flow ·
<code>=&gt;</code> process handoff · <code>[ALIAS:L123]</code> exact source location.
The worker branch is a separate Python process. Repeated calls for new configuration objects and conditional lookups remain explicit.</p>
{startup_html(src)}
<details class="stack-source-map"><summary><strong>Source map</strong> · file aliases used in the tree</summary>
<table><thead><tr><th>Alias</th><th>Complete original file</th></tr></thead><tbody>{''.join(map_rows)}</tbody></table></details>
<div class="stack-tools"><label>Find in tree <input id="tree-search" type="search" placeholder="Function, object or component"></label>
<button id="tree-next" type="button">Next match</button><output id="tree-count" aria-live="polite"></output>
<button id="copy-tree" type="button">Copy all text</button><a href="{TEXT_NAME}" download>Download .txt</a></div>
<p class="stack-scroll-hint">Scroll inside the tree to read long lines. Indentation and source code are preserved.</p>
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
    (task / READER_NAME).write_text(output)
    return len(refs)
