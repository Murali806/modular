"""Build the offline request stories, state machines and OS reference atlas."""

from __future__ import annotations

import hashlib
import html
import json
import re
import textwrap

from call_stack_refs import FILES as STACK_FILES, REFS as STACK_REFS
from os_components_content import CLIENTS_ASCII, CLIENTS_NOTE, COMPONENTS, FLOWS, MACHINES
from serve_story import render_legend, render_stories, story_contract, story_markdown, story_refs

PAGE_NAME = "max_serve_os_components_and_flow.html"


def safe(value):
    return html.escape(str(value), quote=True)


def sources(record):
    return ('<p class="references">' + ' · '.join('{{ref:' + key + '}}' for key in record.get('refs', [])) + '</p>'
            + '<p class="references">' + ' · '.join('{{doc:' + key + '}}' for key in record.get('docs', [])) + '</p>')


def library(key, title, body):
    return f'<details data-reading="{key}"><summary>{safe(title)}</summary><div class="reading-body">{body}</div></details>'


def component_button(key, subtitle=None):
    c = COMPONENTS[key]
    return (f'<button class="component-button" data-card="component:{key}"><strong>{safe(c["name"])}</strong>'
            f'<small>{safe(subtitle or c["role"])}</small><span class="layer">{safe(c["layer"])}</span></button>')


def component_map():
    api = ''.join(component_button(k) for k in ('api', 'loop', 'ready'))
    worker = ''.join(component_button(k) for k in ('worker', 'scheduler', 'native'))
    kernel = ''.join(component_button(k) for k in ('threads', 'epoll', 'socket', 'memory', 'files', 'driver'))
    return f'''<div class="ownership-map">
<div class="process-map"><div class="process-island"><h3>API process <span>HTTP frontend</span></h3>{api}</div>
<div class="ipc-bridge">{component_button('ipc')}<span class="bridge-direction">request → · response ← · cancel →</span></div>
<div class="process-island"><h3>Worker process <span>Model lifetime</span></h3>{worker}</div></div>
<p class="boundary-label">User-space code uses kernel resources and the native device interface</p>
<div class="kernel-map"><h3>Linux services <span>isolation · CPU time · ready I/O · memory · device control</span></h3><div class="kernel-grid">{kernel}</div></div>
<div class="hardware-map">{component_button('nic')}{component_button('gpu')}</div>
</div>'''


def actor_svg(flow):
    width = 40 + 176 * len(flow['actors'])
    positions = {a[0]: 108 + i * 176 for i, a in enumerate(flow['actors'])}
    key = flow['id']
    out = [f'<svg viewBox="0 0 {width} 310" role="group" aria-labelledby="flow-title-{key}">',
           f'<title id="flow-title-{key}">{safe(flow["title"])}</title>',
           f'<defs><marker id="flow-arrow-{key}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" class="arrowhead"/></marker></defs>']
    for actor_info in flow['actors']:
        actor, label, component_key = actor_info[:3]
        layer = actor_info[3] if len(actor_info) > 3 else COMPONENTS[component_key]['layer']
        x = positions[actor]
        out += [f'<g class="actor" data-actor="{actor}" data-card="component:{component_key}" role="button" tabindex="0" aria-label="Explain {safe(label)}">',
                f'<rect x="{x-79}" y="18" width="158" height="81" rx="9" class="actor-box"/>',
                f'<text x="{x}" y="46" text-anchor="middle" class="actor-label">{safe(label)}</text>',
                f'<text x="{x}" y="67" text-anchor="middle" class="actor-subtitle">{safe(layer)}</text>',
                f'<text x="{x}" y="86" text-anchor="middle" class="object-hint">role and interaction ↗</text></g>',
                f'<line x1="{x}" x2="{x}" y1="108" y2="294" class="lifeline"/>']
    for i, e in enumerate(flow['steps']):
        start, end = positions[e['sender']], positions[e['receiver']]
        y = 169 + (i % 3) * 37
        path = f'M{start},{y} h56 v24 h-56' if start == end else f'M{start},{y} H{end}'
        out += [f'<g class="sequence-event{" active" if i == 0 else ""}" data-frame="{i}" aria-hidden="{str(i != 0).lower()}">',
                f'<path d="{path}" data-motion-path class="call-arrow" marker-end="url(#flow-arrow-{key})"/>',
                f'<rect x="{min(start,end)-17}" y="{y-30}" width="34" height="22" rx="6" class="step-number-box"/>',
                f'<text x="{min(start,end)}" y="{y-14}" text-anchor="middle" class="step-number-text">{i+1}</text></g>']
    out.append('<g class="sequence-packet" aria-hidden="true"><circle r="11"/><text text-anchor="middle" y="4">•</text></g></svg>')
    return ''.join(out)


def controls():
    return '''<div class="flow-controls" data-interactive hidden><button data-action="back">Back</button><button data-action="next">Step</button><button data-action="play" aria-pressed="false">Play</button><button data-action="replay">Replay</button><label>Playback <select data-speed><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><output data-progress></output></div>'''


def event_body(e):
    return (f'<h4>Owner</h4><p>{safe(e["owner"])}</p><h4>Interaction</h4><p>{safe(e["interaction"])}</p>'
            f'<h4>State change</h4><p class="result">{safe(e["result"])}</p>' + sources(e))


def render_flows():
    out = []
    for n, flow in enumerate(FLOWS, 1):
        key = flow['id']
        first = flow['steps'][0]
        labels = ''.join(f'<button class="step-label" data-story-step="{i}" data-card="step:{key}:{i}" aria-pressed="{str(i == 0).lower()}"><span>{i+1:02d}</span>{safe(e["title"])}</button>' for i, e in enumerate(flow['steps']))
        explanations = ''.join(library(f'step:{key}:{i}', f'{i+1}. {e["title"]}', event_body(e)) for i, e in enumerate(flow['steps']))
        out.append(f'''<details class="flow-panel" id="flow-{key}" data-flow="{key}"{' open' if n == 1 else ''}>
<summary><span class="section-number">{n:02d}</span><span><strong>{safe(flow['title'])}</strong><small>{safe(flow['intro'])}</small></span></summary>
<div class="flow-body"><div class="diagram-scroll">{actor_svg(flow)}</div>{controls()}
<div class="current-event"><p class="eyebrow">SOURCE / BOUNDARY · event order</p><h3 data-current-title>{safe(first['title'])}</h3><p data-current-owner>{safe(first['owner'])}</p><p data-current-detail>{safe(first['interaction'])}</p><p class="result" data-current-result>{safe(first['result'])}</p><div class="status-strip" data-owner-states></div></div>
<div class="step-list" data-interactive hidden>{labels}</div>
<details class="portable"><summary>Portable high-level sequence</summary><pre>{safe(flow['ascii'])}</pre></details>
<div class="reading-library">{explanations}</div></div></details>''')
    return '\n'.join(out)


def graph_path(machine, edge):
    a, b = machine['states'][edge['start']], machine['states'][edge['end']]
    x1, y1, x2, y2 = a['x'], a['y'], b['x'], b['y']
    if abs(y2-y1) < 5:
        if x2 < x1:
            return f'M{x1},{y1-34} C{x1},{y1-88} {x2},{y2-88} {x2},{y2-34}'
        return f'M{x1+86},{y1} H{x2-86}'
    if abs(x2-x1) < 5:
        return f'M{x1},{y1+34} V{y2-34}'
    if y2 > y1:
        return f'M{x1},{y1+34} C{x1},{y1+80} {x2},{y2-80} {x2},{y2-34}'
    return f'M{x1},{y1-34} C{x1},{y1-90} {x2},{y2+90} {x2},{y2+34}'


def machine_svg(m):
    key = m['id']
    start = next(iter(m['scenarios'].values()))['path'][0]
    out = [f'<svg viewBox="0 0 {m["width"]} {m["height"]}" role="group" aria-labelledby="machine-svg-{key}">',
           f'<title id="machine-svg-{key}">{safe(m["title"])} state machine</title>',
           f'<defs><marker id="state-arrow-{key}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" class="state-arrowhead"/></marker></defs>']
    for e in m['edges']:
        name = e['start'] + '-' + e['end']
        out.append(f'<path d="{graph_path(m,e)}" class="state-edge" data-edge="{name}" marker-end="url(#state-arrow-{key})"><title>{safe(e["label"])}</title></path>')
    for k, s in m['states'].items():
        label = textwrap.wrap(s['label'], 23)
        lines = ''.join(f'<tspan x="{s["x"]}" dy="{0 if i == 0 else 16}">{safe(text)}</tspan>' for i, text in enumerate(label))
        out += [f'<g class="state-node{" active" if k == start else ""}" data-state="{k}" data-card="state:{key}:{k}" role="button" tabindex="0" aria-label="Explain {safe(s["label"])}">',
                f'<rect x="{s["x"]-86}" y="{s["y"]-34}" width="172" height="68" rx="10" class="state-box"/>',
                f'<text x="{s["x"]}" y="{s["y"]-5}" text-anchor="middle" class="state-label">{lines}</text>',
                f'<text x="{s["x"]}" y="{s["y"]+20}" text-anchor="middle" class="state-owner">{safe(s["owner"])}</text></g>']
    out.append('<g class="sequence-packet" aria-hidden="true"><circle r="8"/></g></svg>')
    return ''.join(out)


def render_machines():
    out = []
    for m in MACHINES:
        key = m['id']
        options = ''.join(f'<option value="{k}">{safe(case["label"])}</option>' for k, case in m['scenarios'].items())
        readings = ''.join(library(f'state:{key}:{k}', s['label'], f'<p><strong>Owner:</strong> {safe(s["owner"])}</p><p>{safe(s["meaning"])}</p>'+sources(s)) for k, s in m['states'].items())
        readings += ''.join(library(f'edge:{key}:{e["start"]}-{e["end"]}', e['label'], f'<p>{safe(m["states"][e["start"]]["label"])} → {safe(m["states"][e["end"]]["label"])}</p><p>{safe(e["detail"])}</p>'+sources(e)) for e in m['edges'])
        first_key = next(iter(m['scenarios'].values()))['path'][0]
        first = m['states'][first_key]
        out.append(f'''<article class="machine-panel" id="machine-{key}" data-machine="{key}"><h3>{safe(m['title'])}</h3><p class="muted">{safe(m['intro'])}</p>
<div class="scenario-picker" data-interactive hidden><label>Follow a path <select data-scenario>{options}</select></label></div>
<div class="diagram-scroll state-diagram">{machine_svg(m)}</div>{controls()}
<div class="current-event"><p class="eyebrow">Selected state</p><h4 data-machine-state>{safe(first['label'])}</h4><button class="transition-label" data-current-transition data-card="state:{key}:{first_key}">{safe(first['label'])}</button><p data-machine-detail>{safe(first['meaning'])}</p></div>
<details class="portable"><summary>Portable state machine and meaning</summary><pre>{safe(m['ascii'])}</pre></details><div class="reading-library">{readings}</div></article>''')
    return '\n'.join(out)


def build_os_components(data, task, src, span_finder):
    systems = json.loads((task / 'os_systems_manifest.json').read_text())
    specs = dict(systems['references'])
    for key in ('load_graphs', 'scheduler'):
        specs[key] = {**STACK_REFS[key], 'file': STACK_FILES[STACK_REFS[key]['alias']]}
    used_refs = story_refs('components')
    used_docs = set()
    def collect(record):
        used_refs.update(record.get('refs', [])); used_docs.update(record.get('docs', []))
    for c in COMPONENTS.values(): collect(c)
    for f in FLOWS:
        for e in f['steps']: collect(e)
    for m in MACHINES:
        for s in m['states'].values(): collect(s)
        for e in m['edges']: collect(e)
    research_all = json.loads((task / 'evidence/os_systems_research.json').read_text())
    research = [{k:r[k] for k in ('id','title','url','snapshot','snapshot_sha256')}
                for r in research_all['records'] if r['id'] in used_docs]
    files = {}
    refs = {}
    for key in sorted(used_refs):
        spec = specs[key]
        file = spec['file']
        raw = (task / 'sources' / file).read_bytes()
        files[file] = dict(text=raw.decode(), sha256=hashlib.sha256(raw).hexdigest(), snapshot='sources/'+file)
        start, end = span_finder(raw.decode(), spec['symbol'])
        code = ''.join(raw.decode().splitlines(keepends=True)[start-1:end])
        line = start
        if needle := spec.get('needle'):
            assert code.count(needle) == 1, key
            line += code[:code.index(needle)].count('\n')
        refs[key] = {**spec, 'start':start, 'end':end, 'line':line,
                     'code':code, 'sha256':hashlib.sha256(code.encode()).hexdigest()}
    for m in MACHINES:
        pairs = {(e['start'], e['end']) for e in m['edges']}
        assert len(pairs) == len(m['edges'])
        for case in m['scenarios'].values():
            assert all(k in m['states'] for k in case['path'])
            assert all(pair in pairs for pair in zip(case['path'], case['path'][1:]))
    role_rows = ''.join(f'<tr><td><button class="role-link" data-card="component:{k}">{safe(c["name"])}</button><small>{safe(c["entry"])}</small></td><td>{safe(c["layer"])}</td><td>{safe(c["role"])}</td><td>{safe(c["interacts"])}</td></tr>' for k, c in COMPONENTS.items())
    readings = ''.join(library(f'component:{k}', c['name'], f'<p class="layer">{safe(c["layer"])}</p><p>{safe(c["role"])}</p><h4>Interacts with</h4><p>{safe(c["interacts"])}</p><h4>Back in the call stack</h4><p class="stack-entry">{safe(c["entry"])}</p><ul>'+''.join('<li>'+safe(t)+'</li>' for t in c['details'])+'</ul>'+sources(c)) for k, c in COMPONENTS.items())
    clients = ''.join(f'<tr><td>{safe(name)}</td><td>{safe(owner)}</td><td>{safe(role)}</td></tr>' for name,owner,role in CLIENTS_NOTE)
    template = (src/'os_components.html').read_text()
    for marker, value in {
        'serve-stories':render_stories('components'), 'story-contract':story_contract(), 'serve-legend':render_legend(),
        'component-map':component_map(), 'role-rows':role_rows, 'component-readings':readings,
        'flows':render_flows(), 'machines':render_machines(), 'client-rows':clients,
        'clients-ascii':safe(CLIENTS_ASCII), 'revision':data['revision'],
    }.items(): template=template.replace('{{'+marker+'}}',value)
    def source_link(key):
        r=refs[key]
        return f'<a data-ref="{key}" href="../sources/{r["file"]}#L{r["line"]}">{safe(r["label"])} <small>[{r["alias"]}:L{r["line"]}]</small></a>'
    template=template.replace('{{source-directory}}',''.join('<li>'+source_link(key)+'</li>' for key in refs))
    template=re.sub(r'\{\{ref:([a-z_0-9]+)\}\}',lambda m:source_link(m[1]),template)
    docs={r['id']:r for r in research}
    template=re.sub(r'\{\{doc:([a-z_0-9]+)\}\}',lambda m:f'<a href="{safe(docs[m[1]]["url"])}" target="_blank" rel="noopener">{safe(docs[m[1]]["title"])}</a>',template)
    template=template.replace('{{research}}',''.join(f'<li><a href="{safe(r["url"])}" target="_blank" rel="noopener">{safe(r["title"])}</a> · <a href="../{r["snapshot"]}">saved extract</a></li>' for r in research))
    assert '{{' not in template, 'Unfilled component template'
    payload=dict(revision=data['revision'], scope='One MAX Serve instance: admission and scheduling stories, OS roles and teaching states',
                 components=COMPONENTS, flows=FLOWS, machines=MACHINES, files=files, references=refs, research=research)
    template=template.replace('<!-- COMPONENT_DATA -->','<script id="component-data" type="application/json">'+json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')+'</script>')
    template=template.replace('<!-- COMPONENT_CSS -->','<style>'+(src/'os_components.css').read_text()+'\n'+(src/'serve_story.css').read_text()+'</style>')
    scripts = ('os_components.js', 'serve_story_model.js', 'serve_api_model.js', 'serve_worker_model.js', 'serve_worker.js', 'serve_story.js')
    template=template.replace('<!-- COMPONENT_JS -->',''.join('<script>'+(src/name).read_text()+'</script>' for name in scripts))
    (task/'link_htmls'/PAGE_NAME).write_text(template)
    manifest=dict(revision=data['revision'], page='link_htmls/'+PAGE_NAME,
                  references={k:{a:v for a,v in r.items() if a!='code'} for k,r in refs.items()},
                  files={k:{a:v for a,v in f.items() if a!='text'} for k,f in files.items()},
                  components=list(COMPONENTS), flows={f['id']:len(f['steps']) for f in FLOWS},
                  machines={m['id']:{'states':len(m['states']),'paths':len(m['scenarios'])} for m in MACHINES},
                  research=[{k:r[k] for k in ('id','url','snapshot','snapshot_sha256')} for r in research])
    (task/'os_components_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    md=['# MAX Serve: OS components, roles and high-level flow',
        f'[Interactive focused page](link_htmls/{PAGE_NAME}) · SOURCE revision `{data["revision"]}`.',
        'Linux + NVIDIA. Source calls and external OS/device mechanisms are distinguished. Animation expresses event order.',
        '```text\nUSER SPACE: API + uvloop <-- Unix IPC --> worker + MAX scheduler + native backend\nKERNEL:     processes/threads | sockets/epoll | memory/files | GPU kernel driver\nHARDWARE:   NIC | host RAM/storage | GPU/VRAM\n```',
        *story_markdown('components'), '## Component roles']
    for c in COMPONENTS.values():
        md += [f'### {c["name"]} · {c["layer"]}', c['role'], 'Interacts with: '+c['interacts'], 'Call stack: `'+c['entry']+'`.']
    for f in FLOWS:
        md += ['## '+f['title'], f['intro'], '```text\n'+f['ascii']+'\n```']
        for i,e in enumerate(f['steps'],1): md += [f'### {i}. {e["title"]}', 'Owner: '+e['owner'], e['interaction'], 'State change: '+e['result']]
    for m in MACHINES:
        md += ['## '+m['title'], m['intro'], '```text\n'+m['ascii']+'\n```']
        for s in m['states'].values(): md += [f'- **{s["label"]}** ({s["owner"]}): {s["meaning"]}']
    md += ['## Many clients on this one instance','```text\n'+CLIENTS_ASCII+'\n```',
           '## Evidence and references', '[Source/evidence manifest](os_components_manifest.json)',
           '[Validation record](validation/OS_COMPONENTS_VALIDATION.md)']
    md += [f'- [{r["title"]}]({r["url"]})' for r in research]
    (task/'OS_COMPONENTS_GUIDE.md').write_text('\n\n'.join(md).rstrip()+'\n')
    return len(refs), len(COMPONENTS), sum(len(f['steps']) for f in FLOWS)
