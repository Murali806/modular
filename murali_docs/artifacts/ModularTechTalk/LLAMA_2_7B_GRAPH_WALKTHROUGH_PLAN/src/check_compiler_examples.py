"""Scoped source/hash/navigation validation for the four example readers."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess

TASK=Path(__file__).resolve().parent.parent
REPO=TASK.parents[3]


class Parser(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=[];self.links=[];self.dependencies=[];self.refs=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if 'data-ref' in a:self.refs.append(a['data-ref'])
        if tag in ('script','img','link') and ('src' in a or 'href' in a):self.dependencies.append(a.get('src',a.get('href')))


def check_examples():
    manifest=json.loads((TASK/'compiler_examples_manifest.json').read_text())
    parsers={};file_checks=set();spans=0;links=0
    assert len(manifest['examples'])==4
    workbench=TASK/'link_htmls/max_mlir_mojo_compiler_and_runtime.html'
    home=Parser();home.feed(workbench.read_text())
    parsers[workbench.resolve()]=home
    for e in manifest['examples']:
        page=TASK/e['page'];text=page.read_text();p=Parser();p.feed(text);parsers[page.resolve()]=p
        assert len(p.ids)==len(set(p.ids)),page
        assert not p.dependencies
        payload=json.loads(re.search(r'<script id="compiler-examples-data" type="application/json">(.*?)</script>',text,re.S)[1])
        assert payload['revision']==manifest['revision']
        for key in ('example','references','lessons'):assert payload[key]==e[key]
        assert all(r in payload['references'] for r in p.refs)
        assert Path(e['page']).name in home.links
        for f in e['files']:
            raw=(REPO/f['path']).read_bytes()
            if f['path'] not in file_checks:
                assert raw==subprocess.check_output(['git','show',f"{manifest['revision']}:{f['path']}"],cwd=REPO)
                assert raw==(TASK/'sources'/f['path']).read_bytes();file_checks.add(f['path'])
            assert hashlib.sha256(raw).hexdigest()==f['sha256']
            assert payload['files'][f['path']]['text'].encode()==raw
        for r in payload['references'].values():
            assert r['start']<=r['line']<=r['end']
            raw=''.join(payload['files'][r['file']]['text'].splitlines(keepends=True)[r['start']-1:r['end']]).encode()
            assert hashlib.sha256(raw).hexdigest()==r['sha256'];spans+=1
    for page,p in list(parsers.items()):
        for href in p.links:
            if href.startswith('http'):continue
            name,_,fragment=href.partition('#');target=(page.parent/name).resolve() if name else page
            assert target.is_file(),href
            if fragment and target.suffix=='.html':
                if target not in parsers:
                    parser=Parser();parser.feed(target.read_text());parsers[target]=parser
                assert fragment in parsers[target].ids,href
            links+=1
    return dict(result='PASS',revision=manifest['revision'],pages=4,animations=sum(len(e['lessons']) for e in manifest['examples']),source_files=len(file_checks),source_spans=spans,local_links=links,external_runtime_dependencies=0,evidence='Source/document checks only; no MAX execution')


if __name__=='__main__':
    report=check_examples();(TASK/'validation/compiler_examples_sources.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
