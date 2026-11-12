"""Verify the fresh attention documents against pinned source and local links.

This checks generated documentation only. It does not compile or run MAX.
"""
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

TASK = Path(__file__).resolve().parent.parent
REPO = next(p for p in TASK.parents if (p / '.git').exists())


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = []
        self.links = []
        self.dependencies = []
        self.source_links = []
        self.cards = []
        self.stages = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        a = dict(attributes)
        if 'id' in a:
            self.ids.append(a['id'])
        if tag == 'a' and 'href' in a:
            self.links.append(a['href'])
        if tag in {'script', 'img', 'link', 'iframe'} and ('src' in a or 'href' in a):
            self.dependencies.append(a.get('src', a.get('href')))
        if 'data-source' in a:
            self.source_links.append(a)
        if 'data-card' in a:
            self.cards.append(a['data-card'])
        if 'data-stage' in a:
            self.stages.append(a['data-stage'])


def payload_of(path):
    raw = path.read_text()
    found = re.search(r'<script type="application/json" id="attention-data">(.*?)</script>', raw, re.S)
    assert found, path
    return json.loads(found[1]), Document(raw)


def check_attention():
    manifest = json.loads((TASK / 'attention_walkthrough_manifest.json').read_text())
    file_checks = set()
    local_links = source_links = 0
    pages = []
    parsed = {}
    for item in manifest['pages']:
        path = TASK / item['page']
        data, doc = payload_of(path)
        parsed[path.resolve()] = doc
        assert data['revision'] == manifest['revision']
        assert data['layout'] == manifest['layout'] == 'attention-stage-reader-v1'
        duplicates = [key for key, n in Counter(doc.ids).items() if n > 1]
        assert not duplicates, (path.name, duplicates)
        assert not doc.dependencies, (path.name, doc.dependencies)
        assert doc.stages == [c['id'] for c in data['chapters']]
        assert len(doc.stages) == item['stages']
        cards = {c['id'] for c in data['cards']}
        assert set(doc.cards) <= cards == set(manifest['cards'])
        for card in data['cards']:
            detail = Document(card['html'])
            assert detail.source_links and 'data-animation=' in card['html'], card['id']
            assert not detail.dependencies
        for timeline, frames in data['timelines'].items():
            assert frames, timeline
            assert len({f['mark'] for f in frames}) == len(frames), timeline
            for frame in frames:
                for required in ('title', 'incoming', 'operation', 'outgoing', 'why', 'refs', 'mark'):
                    assert frame[required], (timeline, required)
                for key, line in re.findall(r'\[\[([^:]+):([^\]]+)\]\]', frame['refs']):
                    if key == 'external':
                        assert line in data['external']
                    else:
                        ref = data['references'][key + ':' + line]
                        assert ref['file'] == data['sourcePaths'][key]
        for name, f in data['files'].items():
            assert name == f['path']
            raw = f['text'].encode()
            assert hashlib.sha256(raw).hexdigest() == f['sha256']
            assert (TASK / 'sources' / name).read_bytes() == raw
            if name not in file_checks:
                assert raw == subprocess.check_output(['git', 'show', f"{data['revision']}:{name}"], cwd=REPO)
                file_checks.add(name)
        for source in doc.source_links:
            ref = data['references'][source['data-source']]
            assert source['href'] == f"../sources/{ref['file']}#L{ref['line']}"
            lines = data['files'][ref['file']]['text'].splitlines()
            assert 1 <= ref['line'] <= len(lines)
            assert lines[ref['line'] - 1].strip(), (source['data-source'], 'blank source anchor')
        source_links += len(doc.source_links)
        for href in doc.links:
            split = urlsplit(href)
            if split.scheme or split.netloc:
                assert split.scheme == 'https', href
                continue
            target = (path.parent / unquote(split.path)).resolve() if split.path else path.resolve()
            assert target.exists(), (path.name, href)
            if target.suffix == '.html' and split.fragment:
                if target not in parsed:
                    parsed[target] = Document(target.read_text())
                assert unquote(split.fragment) in parsed[target].ids, (path.name, href)
            local_links += 1
        pages.append(dict(page=item['page'], stages=len(doc.stages), cards=len(cards), source_links=len(doc.source_links)))

    # An independent set of critical call-site anchors. A valid line number
    # alone cannot distinguish the caller from a nearby declaration/comment.
    expected = {
        'entry:74': 'session.compile(graph)',
        'entry:75': 'session.init(compiled)',
        'entry:82': 'model.execute(Q, K, V)',
        'graph:910': 'self._import_kernels',
        'custom:127': 'verify_custom_op',
        'precompile:318': 'runCheckLITPipeline',
        'precompile:322': 'writePrecompiledFile',
        'frontend:381': 'clone()',
        'load:91': 'xxh3',
        'load:160': 'addIfAbsent',
        'load:166': 'lookup',
        'primitives:234': 'kernelPtr',
        'legacy:3643': 'dense_args_addrs',
        'cuda:2262': 'cuModuleLoadDataEx',
        'cuda:2268': 'cuModuleGetFunction',
        'hip:1968': 'hipModuleLoadData',
        'hip:1972': 'hipModuleGetFunction',
    }
    for token, needle in expected.items():
        key, line = token.rsplit(':', 1)
        actual = data['files'][data['sourcePaths'][key]]['text'].splitlines()[int(line) - 1]
        assert needle.lower() in actual.lower(), (token, needle, actual)

    return dict(result='PASS', revision=manifest['revision'], pages=pages,
                source_files=len(file_checks), source_links=source_links,
                local_links=local_links, checked_call_sites=len(expected),
                evidence='Pinned source and document checks; no Bazel, Mojo, MAX or GPU execution')


if __name__ == '__main__':
    report = check_attention()
    output = TASK / 'validation' / 'attention_walkthrough_sources.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
