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
        super().__init__();self.ids=[];self.links=[];self.dependencies=[];self.refs=[];self.originals={};self.original_point=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if 'data-ref' in a:self.refs.append(a['data-ref'])
        if tag=='pre' and 'data-original-point' in a:
            self.original_point=a['data-original-point'];self.originals[self.original_point]=''
        if tag in ('script','img','link') and ('src' in a or 'href' in a):self.dependencies.append(a.get('src',a.get('href')))
    def handle_data(self,data):
        if self.original_point is not None:self.originals[self.original_point]+=data
    def handle_endtag(self,tag):
        if tag=='pre':self.original_point=None


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
        expected_nav=[('commands','00'),('toolchain','01'),('journey','02'),('build','03'),('compile','04'),('run','05')]
        nav=re.findall(r'<a href="#(commands|toolchain|journey|build|compile|run)">(\d{2}) ',text)
        assert nav==expected_nav,(page,nav)
        assert [lesson['id'] for lesson in e['lessons']]==['journey','toolchain','build','compile','run','commands']
        for key,number in expected_nav:
            assert f'data-card="{key}" aria-label="Explain {number}:' in text,(page,key,number)
        toolchain=next(lesson for lesson in e['lessons'] if lesson['id']=='toolchain')
        assert {'bazel_macro','bazel_kernel_sources','package_load','gc_pipeline','llvm_translation','target_info','backend_registry','host_object','nvptx_object','amdgpu_backend'}<=set(toolchain['refs'])
        assert 'kernels/*.mojo is data' in toolchain['ascii'] and 'TargetBackendRegistry lookup' in toolchain['ascii']
        point1=(TASK/'src/compiler_toolchain_point_1.txt').read_text().rstrip('\n')
        originals={'1':point1,**json.loads((TASK/'src/compiler_toolchain_originals.json').read_text())}
        assert p.originals==originals,page
        assert payload['toolchain']['originals']==originals,page
        assert payload['toolchain']['point1_sha256']==hashlib.sha256(point1.encode()).hexdigest()==e['toolchain']['point1_sha256']
        assert all(original in payload['toolchain']['text'] for original in originals.values())
        assert all(f'toolchain-q{n}' in p.ids for n in range(1,8))
        for anchor in ('toolchain-q3-3-1','toolchain-q3-5-2','toolchain-q4-2-2','toolchain-q5-3-1','toolchain-q5-3-2'):
            assert anchor in p.ids,(page,anchor)
        assert 'id="toolchain-animation-tab"' in text and 'id="toolchain-text-tab"' in text
        assert 'data-mode-panel="text"' in text and 'data-mode-panel="animation"' in text
        assert 'ModuleLoad / ExecuteGeneric' in toolchain['ascii'] and 'LLVM TargetRegistry' in toolchain['ascii']
        assert {'mo_to_map','map_to_mogg','mogg_to_mgp','kgen_elaboration','kgen_llvm_passes','llvm_code_gen','package_builder','device_finalization'}<=set(toolchain['refs'])
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
    return dict(result='PASS',revision=manifest['revision'],pages=4,animations=sum(len(e['lessons']) for e in manifest['examples']),questions=7,toolchain_steps=manifest['examples'][0]['toolchain']['steps'],previous_answers_preserved=True,source_files=len(file_checks),source_spans=spans,local_links=links,external_runtime_dependencies=0,evidence='Source/document checks only; no MAX execution')


if __name__=='__main__':
    report=check_examples();(TASK/'validation/compiler_examples_sources.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
