#!/usr/bin/env python3
"""Build only the Llama supplement. Source chunks and assets are shared offline."""
import html
import importlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

import markdown

from catalog import HERE, OUT, PATHS, REPO, REVISION, TASK, load_sources, resolve
from content import CHAPTERS

MAX_FILE = 100 * 1024
CHUNK_CHARS = 28000  # JSON escaping and UTF-8 still fit the per-file limit.


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def md(text):
    return markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])


def source_ref(sources, spec):
    resolved = resolve(sources, spec["file"], spec["needle"], spec.get("occurrence", 1))
    resolved["label"] = spec.get("label") or resolved["label"]
    return resolved


def source_link(r):
    return (f'<a class="source-link" href="sources.html?file={r["file"]}&amp;line={r["line"]}" '
            f'data-source="{r["file"]}" data-line="{r["line"]}">'
            f'{html.escape(r["label"])} <small>L{r["line"]}</small></a>')


def refs_html(refs):
    return '<div class="source-refs">' + " ".join(source_link(r) for r in refs) + '</div>'


def call_tree(c, sources):
    """Link the actual call-tree symbols without changing connector spacing."""
    from tree_links import TREE_LINKS

    links = {}
    for token, spec in TREE_LINKS[c['slug']]:
        assert token in c['tree'], (c['slug'], token)
        links[token] = source_ref(sources, spec)
    pattern = re.compile('|'.join(re.escape(t) for t in sorted(links, key=len, reverse=True)))
    parts, end = [], 0
    for match in pattern.finditer(c['tree']):
        parts.append(html.escape(c['tree'][end:match.start()]))
        r = links[match[0]]
        title = f'{sources[r["file"]]["path"]}:{r["line"]}'
        parts.append(f'<a class="tree-symbol" href="sources.html?file={r["file"]}&amp;line={r["line"]}" '
                     f'data-source="{r["file"]}" data-line="{r["line"]}" '
                     f'title="{html.escape(title)}">{html.escape(match[0])}</a>')
        end = match.end()
    parts.append(html.escape(c['tree'][end:]))
    return ''.join(parts)


def header(title, subtitle, current, nav):
    opts = ''.join(f'<option value="{c["slug"]}.html" {"selected" if c["slug"] == current else ""}>{i:02d} · {html.escape(c["title"])}</option>' for i,c in enumerate(nav))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>{html.escape(title)} · Llama 2 7B</title><link rel="stylesheet" href="reader.css"><link rel="stylesheet" href="windows.css"></head><body>
<a class="skip" href="#main">Skip to lesson</a><header class="site-header"><a class="brand" href="index.html">Llama 2 <b>7B</b></a><label class="jump">Lesson <select id="chapter-select" aria-label="Choose lesson">{opts}</select></label><a href="coverage.html">Coverage</a><a href="sources.html">Source explorer</a><a href="../../max_serve_llama2_7b_startup_call_stack.html">Startup tree ↗</a></header>
<main id="main"><div class="eyebrow">SOURCE · pinned MAX implementation · one device</div><h1>{html.escape(title)}</h1><p class="subtitle">{html.escape(subtitle)}</p>'''


def footer(previous=None, following=None):
    links = (f'<a href="{previous["slug"]}.html">← {html.escape(previous["title"])}</a>' if previous else '<a href="index.html">Course map</a>')
    links += f'<a href="{following["slug"]}.html">{html.escape(following["title"])} →</a>' if following else '<a href="coverage.html">Review the coverage →</a>'
    return f'''<nav class="page-turn">{links}</nav><footer>Source revision <code>{REVISION}</code>. Diagrams are deterministic teaching models, not captured compiler IR or GPU measurements. Full files are available offline. <a href="README.md">Build, scope and verification</a>.</footer></main>
<aside id="detail-card" class="floating-card" hidden aria-label="Explanation"><header class="card-toolbar"><strong id="card-title">Explanation</strong><button data-window-close>Close</button></header><div class="reader-window-content" id="card-body"></div></aside>
<aside id="source-card" class="floating-card source-card" hidden aria-label="Source"><header class="source-toolbar"><strong id="source-title">Source</strong><button data-window-close>Close</button></header><div class="reader-window-content" id="source-body"></div></aside>
<script src="windows.js"></script><script src="sources/manifest.js"></script><script src="model.js"></script><script src="reader.js"></script><script src="labs.js"></script></body></html>'''


def scene_html(c):
    options = '<option value="main">Main route</option>' + ''.join(f'<option value="{s["id"]}">{html.escape(s["title"])}</option>' for s in c['scenarios'])
    return f'''<section class="workbench" id="animation" aria-label="Interactive object and buffer walkthrough">
<div class="scene-top"><button id="lesson-badge" class="lesson-badge" aria-label="Lesson explanation">{c["number"]:02d}</button><strong>{html.escape(c["group"])} · objects, buffers and handoffs</strong><span class="scene-top-right"><button id="fullscreen">Full screen</button><button id="legend" class="legend" aria-label="Component legend">L</button></span></div>
<div class="controls"><button id="reset" title="Reset to the first event">Reset</button><button id="back" title="Previous event">← Back</button><button id="play" aria-pressed="false">Play</button><button id="step">Step →</button><label>Route <select id="scenario">{options}</select></label><label class="seek-label">Event <input id="seek" type="range" min="0" max="{len(c['steps'])-1}" value="0" aria-label="Seek event"></label><output id="progress"></output></div>
<div id="event" class="event" aria-live="polite"><strong></strong><p></p></div><div id="scene" class="scene"><svg class="connections" aria-hidden="true"></svg></div>
<div id="handoff" class="handoff"></div><div id="step-sources"></div><details class="state-detail"><summary>What changed at this event?</summary><div id="delta"></div></details>
<noscript><p>The complete event sequence, call tree and lessons follow below. Enable JavaScript for the movable objects and source windows.</p></noscript></section>'''


def chapter_html(c, nav, i, sources):
    text = header(c['title'], c['subtitle'], c['slug'], nav)
    if c['slug'] == 'index':
        text += '<div class="route-note"><b>Serving gate:</b> the saved checkpoint uses FP16, but this revision’s Llama architecture does not list <code>float16</code> as a supported encoding. The lessons follow a compatible resolved unquantized configuration; they do not claim that the original checkpoint was served. <a href="build_startup.html#q2">Inspect the validation gate →</a></div>'
    text += '<nav class="in-page"><a href="#call-tree">Call order</a><a href="#animation">Animation</a><a href="#deep-dive">Detailed explanation</a><a href="#event-transcript">All events / print</a></nav>'
    text += f'<details class="call-tree" id="call-tree" open><summary>Source-ordered call tree · select a linked symbol to inspect its source</summary><pre>{call_tree(c, sources)}</pre>{refs_html(c["refs"])}</details>'
    text += scene_html(c)
    if c.get('lab'):
        text += f'<section class="lab" data-lab="{c["lab"]}"><h2>Change the illustrative inputs</h2><div id="lab-controls"></div><div id="lab-result" aria-live="polite"></div></section>'
    if c['slug'] == 'index':
        group = None
        text += '<section class="course-map"><h2>Follow the model, then follow its execution</h2>'
        for n,p in enumerate(nav[1:],1):
            if group != p['group']:
                if group is not None: text += '</div>'
                group = p['group']; text += f'<h3>{html.escape(group)}</h3><div class="lesson-grid">'
            text += f'<a class="lesson-tile" href="{p["slug"]}.html"><span>{n:02d}</span><b>{html.escape(p["title"])}</b><small>{html.escape(p["subtitle"])}</small></a>'
        text += '</div></section>'
    text += '<section id="deep-dive"><h2>Detailed explanation</h2>'
    for n,s in enumerate(c['sections'],1):
        text += f'<details class="lesson-section" id="q{n}" {"open" if s["open"] else ""}><summary><span>{c["number"]}.{n}</span> {html.escape(s["title"])}</summary><div class="section-body">{md(s["text"])}{refs_html(s["refs"])}</div></details>'
    text += '</section><details id="event-transcript" class="transcript"><summary>Complete event transcript · portable / no JavaScript</summary>'
    for route in [dict(title="Main route",steps=c['steps'])] + c['scenarios']:
        text += f'<h3>{html.escape(route["title"])}</h3><ol>'
        for s in route['steps']:
            text += f'<li><b>{html.escape(s["title"])}</b><p>{html.escape(s["detail"])}</p><dl>'
            for aid,state in s['changes'].items():
                values = [x['label'] for x in state['slots']] + state['facts']
                text += f'<dt>{html.escape(aid)} · {html.escape(state["state"])}</dt><dd>{html.escape(" · ".join(values))}</dd>'
            text += '</dl>' + refs_html(s['refs']) + '</li>'
        text += '</ol>'
    text += '</details>'
    data = {k:v for k,v in c.items() if k not in ('sections','tree')}
    text += '<script type="application/json" id="lesson-data">' + packed(data) + '</script>'
    return text + footer(nav[i-1] if i else None,nav[i+1] if i+1<len(nav) else None)


def build_sources(sources):
    target = OUT/'sources'; target.mkdir(exist_ok=True)
    manifest = {}
    for alias, info in sources.items():
        content = info['text']
        chunks = [content[i:i+CHUNK_CHARS] for i in range(0,len(content),CHUNK_CHARS)]
        files = []
        for i, chunk in enumerate(chunks):
            filename = f'{alias}_{i:02d}.js'; files.append('sources/'+filename)
            (target/filename).write_text('window.LLAMA_CHUNKS=window.LLAMA_CHUNKS||{};window.LLAMA_CHUNKS['+packed(alias+':'+str(i))+']='+packed(chunk)+';\n')
        manifest[alias] = {k:v for k,v in info.items() if k != 'text'}
        manifest[alias].update(chunks=files, lines=len(content.splitlines()), bytes=len(content.encode()))
    (target/'manifest.js').write_text('window.LLAMA_SOURCES='+packed(manifest)+';\n')
    return manifest


def build():
    for filename in sorted(HERE.glob('chapters_*.py')): importlib.import_module(filename.stem)
    sources = load_sources(); OUT.mkdir(exist_ok=True)
    assert len({c['slug'] for c in CHAPTERS}) == len(CHAPTERS)
    for i,c in enumerate(CHAPTERS):
        c['number'] = i
        c['refs'] = [source_ref(sources,s) for s in c['refs']]
        for s in c['sections']: s['refs'] = [source_ref(sources,r) for r in s['refs']]
        for route in [c]+c['scenarios']:
            for s in route['steps']:
                s['refs'] = [source_ref(sources,r) for r in s['refs']]
                assert s['changes'] and s['refs'], (c['slug'],s['title'])
    manifest = build_sources(sources)
    for i,c in enumerate(CHAPTERS): (OUT/(c['slug']+'.html')).write_text(chapter_html(c,CHAPTERS,i,sources))
    for name in ('reader.css','reader.js','model.js','labs.js','README.md'):
        shutil.copyfile(HERE/name,OUT/name)
    for ext in ('js','css'):
        # Reuse the accepted card implementation without duplicating authoring code.
        path=f'murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/compiler_card_windows.{ext}'
        data=subprocess.check_output(['git','show','cde7f11ef5^:'+path],cwd=REPO)
        (OUT/f'windows.{ext}').write_bytes(data)
    index = header('Complete source explorer','Local source files; exact line links; no network requests.','',CHAPTERS)
    index += '<section id="source-explorer"><label>File <select id="source-select">'+''.join(f'<option value="{a}">{html.escape(v["path"])}</option>' for a,v in manifest.items())+'</select></label><label>Find in this file <input id="source-search" placeholder="symbol or text"></label><button id="source-find">Find next</button><div id="source-view"></div></section><noscript><p>The lessons and source references remain readable without scripting. Original complete source files:</p><ul>'
    for a,v in manifest.items():
        path=REPO/v['path']
        if not path.exists(): path=TASK/'sources'/v['path']
        if path.exists(): index+=f'<li><a href="{html.escape(os.path.relpath(path,OUT))}">{html.escape(v["path"])}</a></li>'
    (OUT/'sources.html').write_text(index+'</ul></noscript>'+footer())
    (OUT/'chapters.json').write_text(json.dumps([dict(slug=c['slug'],title=c['title'],group=c['group'],events=len(c['steps']),sections=len(c['sections'])) for c in CHAPTERS],indent=2)+'\n')
    from coverage import build_coverage
    build_coverage(CHAPTERS,header,footer)
    for path in OUT.rglob('*'):
        if path.is_file() and path.stat().st_size > MAX_FILE:
            raise ValueError(f'{path.name} exceeds 100 KiB: {path.stat().st_size}')
    files=[p for p in OUT.rglob('*') if p.is_file()]
    print(f'{len(CHAPTERS)} lessons; {len(sources)} complete sources; {len(files)} files; {sum(p.stat().st_size for p in files):,} bytes; largest {max(p.stat().st_size for p in files):,} bytes')


if __name__ == '__main__': build()
