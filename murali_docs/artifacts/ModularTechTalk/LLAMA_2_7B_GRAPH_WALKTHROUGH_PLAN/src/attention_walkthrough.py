"""Build the fresh, stage-led attention walkthrough and its GPU companion."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

from cells import REVISION
from attention_walkthrough_content import PAGE, DRIVER_PAGE, SOURCES, CHAPTERS
from attention_walkthrough_cards import CARDS
from attention_driver_content import CHAPTERS as DRIVER_CHAPTERS, EXTERNAL
from attention_walkthrough_steps import TIMELINES

SRC = Path(__file__).resolve().parent
TASK = SRC.parent
REPO = next(p for p in TASK.parents if (p / '.git').exists())

SOURCE_PATHS = dict(SOURCES, **{
    'context': 'max/python/max/_mlir_context.py',
    'builtin': 'max/python/max/_core/max_license/modules/dialects/builtin.cpp',
    'graph-op': 'GraphCompiler/lib/MODialect/MOOps.cpp',
    'type': 'max/python/max/graph/type.py',
    'moto-map': 'GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp',
    'mapto-mogg': 'GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp',
    'moggto-mgp': 'GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp',
})

CARD_TITLES = {c['id']: c['title'] for c in CARDS}
TAG = re.compile(r'\[\[([^:\]]+):([^\]]+)\]\]|\{\{([a-z-]+)\}\}')


def colored(text):
    value = html.escape(text)
    return re.sub(r'\b(P1|G1|IR1|S1|C1|M1|A1|D1|H1|KD1|MD1|O1|GPU0|CTX1)\b|\b([A-Za-z_][\w.:]*(?=\())',
                  lambda m: '<strong class="'+('identity' if m[1] else 'emphasis')+'">'+m[0]+'</strong>', value)


class Renderer:
    def __init__(self, files):
        self.files = files
        self.refs = {}

    def reference(self, key, line):
        path = SOURCE_PATHS[key]
        n = int(line)
        lines = self.files[path]['text'].splitlines()
        if not 1 <= n <= len(lines):
            raise ValueError(f'Out-of-bounds source link: {key}:{line}')
        token = f'{key}:{n}'
        ext = Path(path).suffix
        alias = {'.py':'PY','.cpp':'C++','.h':'C++','.mojo':'MOJO','.md':'DOC'}.get(ext, 'BUILD')
        ref = dict(id=token, file=path, line=n, start=1, end=len(lines),
                   label=f'{Path(path).name}:{n}', alias=alias,
                   language={'.py':'python','.cpp':'cpp','.h':'cpp','.mojo':'mojo'}.get(ext,'text'),
                   sha256=self.files[path]['sha256'])
        self.refs[token] = ref
        title = html.escape(f'{path}:{n} — {lines[n-1].strip()}', quote=True)
        return (f'<a class="source-link" data-source="{token}" href="../sources/{path}#L{n}" '
                f'title="{title}">↗ {html.escape(Path(path).name)}:{n}</a>')

    def token(self, match):
        key, line, card = match.groups()
        if card:
            if card not in CARD_TITLES:
                raise ValueError(f'Unknown card: {card}')
            label = CARD_TITLES[card]
            return (f'<a class="card-link" href="#note-{card}" data-card="{card}" '
                    f'aria-haspopup="dialog">{html.escape(label)} <span aria-hidden="true">＋</span></a>')
        if key == 'driver-guide':
            return (f'<a class="deep-link" href="{DRIVER_PAGE}#stage-{line}">'
                    'Open the GPU artifact / ABI / driver deep dive ↗</a>')
        if key == 'external':
            value = EXTERNAL[line]
            return (f'<a class="source-link external" href="{value["url"]}" target="_blank" '
                    f'rel="noopener">↗ {html.escape(value["title"])}</a>')
        if key not in SOURCE_PATHS:
            raise ValueError(f'Unknown source: {key}')
        return self.reference(key, line)

    def inline(self, text, *, prose=False):
        result = []
        cursor = 0
        # Tokens are expanded after escaping prose, so external/source content
        # cannot introduce HTML or scripts.
        for match in TAG.finditer(text):
            result.append(self.styled(text[cursor:match.start()], prose))
            result.append(self.token(match))
            cursor = match.end()
        result.append(self.styled(text[cursor:], prose))
        return ''.join(result)

    @staticmethod
    def styled(text, prose):
        if not prose:
            return colored(text)
        chunks = re.split(r'(`[^`]+`|\*\*[^*]+\*\*)', text)
        result = []
        for c in chunks:
            if c.startswith('`') and c.endswith('`'):
                result.append('<code>' + colored(c[1:-1]) + '</code>')
            elif c.startswith('**') and c.endswith('**'):
                result.append('<strong class="emphasis">' + colored(c[2:-2]) + '</strong>')
            else:
                result.append(colored(c))
        return ''.join(result)

    def tree(self, text):
        rows = []
        for line in text.splitlines():
            prefix = re.match(r'^[│ ├└─]*', line)[0]
            body = line[len(prefix):]
            root = bool(body.startswith('caller:'))
            rows.append(f'<div class="tree-row{" tree-root" if root else ""}">'
                        f'<span class="tree-branch" aria-hidden="true">{html.escape(prefix) or " "}</span>'
                        f'<span class="tree-content">{self.inline(body)}</span></div>')
        return '<div class="call-tree" aria-label="Source-ordered call tree">' + ''.join(rows) + '</div>'

    def markdown(self, text):
        lines = text.strip().splitlines()
        parts = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            if line.startswith('```'):
                content = []
                i += 1
                while i < len(lines) and not lines[i].startswith('```'):
                    content.append(lines[i]); i += 1
                parts.append('<pre class="code-example"><code>' + colored('\n'.join(content)) + '</code></pre>')
                i += 1
                continue
            if line.startswith('|'):
                rows = []
                while i < len(lines) and lines[i].startswith('|'):
                    if not re.fullmatch(r'[| :\-]+', lines[i]):
                        rows.append([self.inline(c.strip(), prose=True) for c in lines[i].strip('|').split('|')])
                    i += 1
                table = '<div class="table-scroll"><table><thead><tr>'
                table += ''.join(f'<th>{c}</th>' for c in rows[0]) + '</tr></thead><tbody>'
                table += ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows[1:])
                parts.append(table + '</tbody></table></div>')
                continue
            if line.startswith('#'):
                value = line.lstrip('#').strip()
                parts.append('<h3>' + self.inline(value, prose=True) + '</h3>')
                i += 1
                continue
            para = [line]; i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#','```','|')):
                para.append(lines[i]); i += 1
            parts.append('<p>' + self.inline(' '.join(para), prose=True) + '</p>')
        return ''.join(parts)

    def player(self, key, title):
        frames = TIMELINES.get(key, [])
        transcript = ''.join(
            '<li><strong>' + html.escape(f['title']) + '</strong><p>' +
            self.inline(f'IN: {f["incoming"]} → {f["operation"]} OUT: {f["outgoing"]} {f["refs"]}') +
            '</p><p class="reason">Why: ' + self.inline(f['why']) + '</p></li>' for f in frames)
        return f'''<div class="player" data-animation="{key}" aria-label="{html.escape(title)} animation">
<div class="player-controls" data-interactive><button type="button" data-action="play" aria-pressed="false">Play</button><button type="button" data-action="back">Back</button><button type="button" data-action="step">Step</button><button type="button" data-action="reset">Reset</button><label class="seek-label">Step <input type="range" min="0" max="{max(0,len(frames)-1)}" value="0" data-seek aria-label="Animation step"></label><output data-count>1 / {len(frames)}</output><button type="button" class="legend-button" data-legend aria-label="Open diagram legend">L</button></div>
<div class="event-line" aria-live="polite" aria-atomic="true"><span class="step-number" data-step-number>01</span><div><strong data-event-title>{html.escape(frames[0]['title'] if frames else 'Follow the arithmetic')}</strong><span data-event-action>{html.escape(frames[0]['operation'] if frames else 'Small independently calculated example')}</span></div></div>
<div class="canvas-scroll" tabindex="0" aria-label="Animation canvas; scroll horizontally on a narrow screen"><svg class="scene" viewBox="0 0 1040 510" role="img" aria-label="{html.escape(title)}; schematic source-based diagram"><title>{html.escape(title)}</title><defs><marker id="arrow-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor"/></marker></defs><g data-scene-background></g><g data-scene-edges></g><g data-scene-actors></g></svg></div>
<div class="io-strip"><div><span>Input</span><p data-in></p></div><div><span>Result</span><p data-out></p></div><div><span>Why this step</span><p data-why></p></div></div>
<div class="frame-evidence" data-evidence></div>
<details class="portable" open><summary>Portable flow · every step and its source</summary><ol>{transcript}</ol></details>
</div>'''

    def render_chapter(self, c, index, total, companion):
        identity = html.escape(c['number'])
        call = self.inline(c['call'])
        tree = self.tree(c['tree'])
        commands = ''
        if c['id'] == 'build':
            commands = '<div class="commands"><span>From the repository root</span><pre><code>' + colored(
                './bazelw build //max/examples/custom_ops:fused_attention\n'
                './bazelw run //max/examples/custom_ops:fused_attention\n'
                './bazelw test //max/examples/custom_ops:fused_attention.example-test') + '</code></pre></div>'
        guide = ''
        if not companion and c['id'] in {'compile','init','execute'}:
            anchor = {'compile':'artifacts','init':'host-link','execute':'launch'}[c['id']]
            guide = f'<p class="chapter-link">{self.inline("[[driver-guide:"+anchor+"]]")}</p>'
        return f'''<section class="chapter" id="stage-{c['id']}" data-stage="{c['id']}" aria-labelledby="title-{c['id']}">
<header class="chapter-header"><div><span class="eyebrow">{identity} · {'GPU artifact deep dive' if companion else 'Source-ordered walkthrough'}</span><h2 id="title-{c['id']}">{html.escape(c['title'])}</h2></div><button type="button" data-fullscreen data-interactive aria-label="Toggle fullscreen for this stage">Fullscreen</button></header>
<p class="chapter-summary">{self.inline(c['summary'])}</p><p class="call-expression"><code>{call}</code></p>
<div class="view-tabs" role="tablist" aria-label="{html.escape(c['title'])} views" data-interactive>{''.join(f'<button type="button" role="tab" id="tab-{c["id"]}-{v}" data-view="{v}" aria-controls="view-{c["id"]}-{v}" aria-selected="{str(v=="tree").lower()}">{label}</button>' for v,label in [('tree','Call Stack Tree'),('animation','Animation'),('why','Why this design')])}</div>
<div class="view tree-view" id="view-{c['id']}-tree" role="tabpanel" aria-labelledby="tab-{c['id']}-tree" data-view-panel="tree">{commands}{tree}{guide}</div>
<div class="view animation-view" id="view-{c['id']}-animation" role="tabpanel" aria-labelledby="tab-{c['id']}-animation" data-view-panel="animation">{self.player(c['id'],c['title'])}</div>
<div class="view why-view" id="view-{c['id']}-why" role="tabpanel" aria-labelledby="tab-{c['id']}-why" data-view-panel="why">{self.markdown(c['why'])}{guide}</div>
<footer class="chapter-footer" data-interactive><button type="button" data-prev-stage {'disabled' if index==0 else ''}>← Previous stage</button><span>{index+1} / {total}</span><button type="button" data-next-stage {'disabled' if index==total-1 else ''}>Next stage →</button></footer>
</section>'''


def write_if_changed(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != value:
        path.write_text(value)


def collect_sources():
    files = {}
    for path in sorted(set(SOURCE_PATHS.values())):
        raw = (REPO / path).read_bytes()
        expected = subprocess.check_output(['git','show',f'{REVISION}:{path}'], cwd=REPO)
        if raw != expected:
            raise ValueError(f'Source differs from pinned revision: {path}')
        files[path] = dict(path=path, text=raw.decode(), sha256=hashlib.sha256(raw).hexdigest())
        write_if_changed(TASK / 'sources' / path, raw.decode())
    return files


def build_attention(example):
    files = collect_sources()
    renderer = Renderer(files)
    # Render the source-linked prose first so every reference is available to
    # browser animations before serializing the payload.
    card_html = {}
    for c in CARDS:
        math_control = ('<label class="math-choice">Query row <select data-query>'
                        '<option value="0">q₀ = [1,0]</option><option value="1">q₁ = [0,1]</option>'
                        '</select></label>') if c['id']=='attention-math' else ''
        card_html[c['id']] = (f'<p class="card-intro">{renderer.inline(c["intro"])}</p>' +
             renderer.tree(c['tree']) + '<details class="card-motion"><summary>Watch this mechanism step by step</summary>' +
             math_control + renderer.player(c['animation'], c['title']) + '</details>' + renderer.markdown(c['body']))
    for frames in TIMELINES.values():
        for f in frames:
            f['source_html'] = renderer.inline(f['refs'])
    nav_names = {'build':'Build','graph':'Graph · G1','session':'Session · S1','compile':'Compile · C1',
                 'init':'Initialize · M1','execute':'Execute','artifacts':'Artifacts','host-link':'Host linking',
                 'device-load':'Device loading','launch':'ABI / dispatch','completion':'Completion'}
    all_chapters = [(CHAPTERS, PAGE, False), (DRIVER_CHAPTERS, DRIVER_PAGE, True)]
    pages = []
    for chapters, filename, companion in all_chapters:
        sections = ''.join(renderer.render_chapter(c,i,len(chapters),companion) for i,c in enumerate(chapters))
        note_bank = '<section class="note-bank" aria-label="Focused explanation notes">' + ''.join(
            f'<details id="note-{c["id"]}"><summary>{html.escape(c["title"])}</summary>{card_html[c["id"]]}</details>' for c in CARDS) + '</section>'
        nav = ''.join(f'<a href="#stage-{c["id"]}" data-stage-link="{c["id"]}"><span>{html.escape(c["number"])}</span><strong>{nav_names[c["id"]]}</strong></a>' for c in chapters)
        snapshot_links = ''.join(f'<li><a href="../sources/{html.escape(path)}">{html.escape(path)}</a></li>' for path in files)
        original_sources = ''.join('<details><summary>' + html.escape(path) + ' · complete unchanged source</summary><pre class="original-source"><code>' + html.escape(files[path]['text']) + '</code></pre></details>' for path in [SOURCES['entry'],SOURCES['kernel']])
        refs_html = ''.join(f'<li><a href="{v["url"]}" target="_blank" rel="noopener">{html.escape(v["title"])}</a><p>{html.escape(v["note"])}</p></li>' for v in EXTERNAL.values())
        payload = dict(revision=REVISION, layout='attention-stage-reader-v1', companion=companion,
                       chapters=[dict(id=c['id'],number=c['number'],title=c['title']) for c in chapters],
                       cards=[dict(id=c['id'],title=c['title'],html=card_html[c['id']]) for c in CARDS],
                       references=renderer.refs, files=files, timelines=TIMELINES,
                       sourcePaths=SOURCE_PATHS, external=EXTERNAL)
        title = 'GPU artifacts, ABI and driver calls' if companion else 'Fused attention, from build to GPU execution'
        subtitle = ('Follow compiled bytes into host callables and GPU handles, then follow arguments and completion.' if companion else
                    'One stage at a time. Follow the actual GPU branch of fused_attention.py, with exact source links and a step-by-step animation at every boundary.')
        back = (f'<a href="{PAGE}#stage-init">← Attention walkthrough</a>' if companion else
                '<a href="max_example_01_add_one_compilation_and_execution.html">← Add-one reference</a>')
        values = dict(title=title, subtitle=subtitle, back=back, nav=nav, chapters=sections,
                      note_bank=note_bank, snapshots=snapshot_links, original_sources=original_sources,
                      research=refs_html, revision=REVISION, identity=('A1 → H1 · D1 → KD1 · data → O1' if companion else 'P1 → G1 · S1 → C1 → M1 → O1'),
                      payload=json.dumps(payload, ensure_ascii=False).replace('<','\\u003c'),
                      css=(SRC/'attention_walkthrough.css').read_text()+'\n'+(SRC/'compiler_card_windows.css').read_text(),
                      js='\n'.join((SRC/p).read_text() for p in ['compiler_card_windows.js','attention_walkthrough_model.js','attention_walkthrough_visuals.js','attention_walkthrough.js']))
        shell = (SRC/'attention_walkthrough.html').read_text()
        tokens = set(re.findall(r'\{\{([a-z_]+)\}\}',shell))
        if tokens != values.keys():
            raise ValueError(f'Unexpected template tokens: {tokens ^ values.keys()}')
        page = re.sub(r'\{\{([a-z_]+)\}\}',lambda m:values[m[1]],shell)
        write_if_changed(TASK/'link_htmls'/filename,page)
        pages.append(dict(page='link_htmls/'+filename, stages=len(chapters),
                          steps=sum(len(TIMELINES[c['id']]) for c in chapters)))
        print(f'Built {filename}: {len(chapters)} stages, {len(CARDS)} focused notes')
    manifest = dict(revision=REVISION, layout='attention-stage-reader-v1', pages=pages,
                    references=renderer.refs,
                    files=[{k:v for k,v in f.items() if k!='text'} for f in files.values()],
                    cards=[c['id'] for c in CARDS],
                    evidence='SOURCE; schematic diagrams and browser arithmetic; no MAX or GPU execution')
    write_if_changed(TASK/'attention_walkthrough_manifest.json',json.dumps(manifest,indent=2)+'\n')
    return dict(page='link_htmls/'+PAGE, example=example, layout=manifest['layout'],
                references=renderer.refs, files=manifest['files'],
                lessons=[dict(id=c['id'],title=c['title'],number=c['number']) for c in CHAPTERS],
                toolchain=dict(reading_views=['tree','animation','why'], cards=manifest['cards']),
                companion='link_htmls/'+DRIVER_PAGE)
