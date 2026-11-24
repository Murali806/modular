#!/usr/bin/env python3
"""Check provenance, coverage and portable navigation without running MAX."""
import hashlib
from html.parser import HTMLParser
import importlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import parse_qs, unquote, urlsplit

from catalog import HERE, OUT, REPO, REVISION, TASK, load_sources, resolve
from content import CHAPTERS
from coverage import MAPPING
from tree_links import TREE_LINKS


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.path, self.ids, self.links, self.refs = path, set(), [], []
        self.tree_links = 0
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            assert a['id'] not in self.ids, (self.path.name, 'duplicate id', a['id'])
            self.ids.add(a['id'])
        for key in ('href', 'src'):
            if a.get(key):
                self.links.append((tag, key, a[key]))
        if 'data-source' in a:
            self.refs.append((a['data-source'], int(a['data-line'])))
        if 'tree-symbol' in a.get('class', '').split():
            self.tree_links += 1


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    for file in sorted(HERE.glob('chapters_*.py')):
        importlib.import_module(file.stem)
    sources = load_sources()
    manifest_text = (OUT/'sources/manifest.js').read_text()
    manifest = json.loads(manifest_text.removeprefix('window.LLAMA_SOURCES=').rstrip(';\n'))
    assert set(manifest) == set(sources)
    provenance = {'git_revision': 0, 'saved_snapshot': 0}
    for alias, spec in manifest.items():
        chunks = []
        for i, filename in enumerate(spec['chunks']):
            script = (OUT/filename).read_text()
            key = json.dumps(f'{alias}:{i}')
            prefix = f'window.LLAMA_CHUNKS=window.LLAMA_CHUNKS||{{}};window.LLAMA_CHUNKS[{key}]='
            assert script.startswith(prefix), filename
            chunks.append(json.loads(script[len(prefix):].rstrip(';\n')))
        text = ''.join(chunks)
        assert text == sources[alias]['text'], (alias, 'incomplete or changed source chunks')
        assert sha(text.encode()) == spec['sha256'], alias
        assert len(text.splitlines()) == spec['lines'], alias
        pinned = subprocess.run(['git', 'show', f'{REVISION}:{spec["path"]}'],
                                cwd=REPO, capture_output=True)
        if pinned.returncode == 0:
            assert sha(pinned.stdout) == spec['sha256'], (alias, 'source revision drift')
            provenance['git_revision'] += 1
        else:
            snapshot = TASK/'sources'/spec['path']
            assert snapshot.is_file() and sha(snapshot.read_bytes()) == spec['sha256'], alias
            # Compare with the tracked saved snapshot as an independent input.
            saved = subprocess.check_output(['git', 'show', f'HEAD:{snapshot.relative_to(REPO)}'], cwd=REPO)
            assert sha(saved) == spec['sha256'], (alias, 'saved snapshot drift')
            provenance['saved_snapshot'] += 1

    anchors = states = 0
    slugs = {c['slug'] for c in CHAPTERS}
    assert slugs == set(TREE_LINKS)
    for c in CHAPTERS:
        actors = {a['id'] for a in c['actors']}
        assert len(actors) == len(c['actors']), c['slug']
        assert all(a['owner'] and a['kind'] and a['purpose'] for a in c['actors'])
        specs = c['refs'] + [r for section in c['sections'] for r in section['refs']]
        for token, r in TREE_LINKS[c['slug']]:
            assert token in c['tree'], (c['slug'], token)
            specs.append(r)
        for route in [dict(id='main', steps=c['steps'])] + c['scenarios']:
            state = {a: {'state': '', 'slots': [], 'facts': []} for a in actors}
            for event in route['steps']:
                assert event['changes'] and event['refs'], (c['slug'], event['title'])
                assert set(event['changes']) <= actors
                assert all(len(edge) == 2 and set(edge) <= actors for edge in event['edges'])
                if event['transfer']:
                    assert {event['transfer'][0], event['transfer'][-1]} <= actors
                state.update(event['changes'])
                identities = [slot['id'] for value in state.values() for slot in value['slots']]
                assert len(identities) == len(set(identities)), (c['slug'], route['id'], event['title'])
                specs.extend(event['refs'])
                states += 1
        for r in specs:
            resolved = resolve(sources, r['file'], r['needle'], r.get('occurrence', 1))
            assert r['needle'] in sources[r['file']]['text'].splitlines()[resolved['line']-1]
            anchors += 1

    observed = {p.stem.removeprefix('max_example_04_') for p in (TASK/'link_htmls').glob('max_example_04*.html')}
    assert observed == set(MAPPING)
    coverage = json.loads((OUT/'coverage.json').read_text())
    assert {r['example'] for r in coverage} == {f'max_example_04_{key}.html' for key in MAPPING}
    assert all(set(r['lessons']) <= slugs for r in coverage)

    pages = {p.resolve(): Page(p) for p in OUT.glob('*.html')}
    links = inline_refs = 0
    for path, page in list(pages.items()):
        if path.stem in slugs:
            assert page.tree_links >= 2, path.name
            assert {'call-tree', 'animation', 'deep-dive', 'event-transcript'} <= page.ids
        for alias, line in page.refs:
            assert alias in manifest and 1 <= line <= manifest[alias]['lines'], (path.name, alias, line)
            inline_refs += 1
        for tag, attr, href in page.links:
            parsed = urlsplit(href)
            assert not parsed.scheme and not parsed.netloc, (path.name, 'nonportable link', href)
            target = (path.parent/unquote(parsed.path)).resolve() if parsed.path else path
            assert target.is_file(), (path.name, 'missing target', href)
            if parsed.fragment:
                if target not in pages:
                    pages[target] = Page(target)
                assert unquote(parsed.fragment) in pages[target].ids, (path.name, 'missing fragment', href)
            if target.name == 'sources.html' and parsed.query:
                query = parse_qs(parsed.query)
                alias, line = query['file'][0], int(query['line'][0])
                assert alias in manifest and 1 <= line <= manifest[alias]['lines']
            links += 1

    text_path = TASK/'MAX_SERVE_LLAMA2_7B_STARTUP_CALL_STACK.txt'
    startup = (TASK/'max_serve_llama2_7b_startup_call_stack.html').read_text()
    stack = json.loads(re.search(r'<script\b[^>]*id="stack-data"[^>]*>(.*?)</script>', startup, re.S)[1])
    assert stack['text'] == text_path.read_text(), 'startup embedded text differs from TXT'
    new_link = 'link_htmls/llama_2_7b/index.html'
    assert startup.count(new_link) == 3 and text_path.read_text().count(new_link) == 1
    assert re.search(r'<a class="tree-doc-link"[^>]*href="'+re.escape(new_link)+'"', startup)

    files = [p for p in OUT.rglob('*') if p.is_file()]
    assert all(p.stat().st_size <= 100*1024 for p in files), 'generated file exceeds 100 KiB'
    assert not any(p.suffix in {'.png', '.pdf', '.mef', '.bin', '.safetensors'} for p in files)
    report = dict(lessons=len(CHAPTERS), routes=len(CHAPTERS)+sum(len(c['scenarios']) for c in CHAPTERS),
                  states=states, author_anchors=anchors, rendered_source_links=inline_refs,
                  local_links=links, complete_sources=len(sources), provenance=provenance,
                  example_04_companions=len(coverage), files=len(files),
                  total_bytes=sum(p.stat().st_size for p in files),
                  largest_file_bytes=max(p.stat().st_size for p in files),
                  largest_html_bytes=max(p.stat().st_size for p in OUT.glob('*.html')))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
