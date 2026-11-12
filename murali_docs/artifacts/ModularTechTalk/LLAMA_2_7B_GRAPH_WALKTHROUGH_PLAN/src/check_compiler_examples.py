"""Scoped source/hash/navigation validation for the four example readers."""
from collections import Counter
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
        self.original_points=[];self.original_fragments=[];self.original_fragment=None;self.diagrams=[];self.presentation=[];self.suppressed_tag=None
        self.caller_trees=[];self.call_tree_ids=[];self.attention_scenarios={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if 'data-ref' in a:self.refs.append(a['data-ref'])
        if 'data-caller-source' in a:self.caller_trees.append(a.get('data-caller-tree'))
        if 'data-call-tree-id' in a:self.call_tree_ids.append(a['data-call-tree-id'])
        if 'data-am-scenarios' in a:self.attention_scenarios[a['data-am-lesson']]=json.loads(a['data-am-scenarios'])
        if 'data-original-point' in a:self.original_points.append(a['data-original-point'])
        if tag=='pre' and 'data-original-point' in a:
            self.original_point=a['data-original-point'];self.originals[self.original_point]=''
        if tag=='pre' and 'data-original-fragment' in a:
            self.original_fragment=len(self.original_fragments);self.original_fragments.append('')
        if tag=='figure' and 'data-diagram' in a:self.diagrams.append(a['data-diagram'])
        if tag in ('script','style'):self.suppressed_tag=tag
        if tag in ('script','img','link') and ('src' in a or 'href' in a):self.dependencies.append(a.get('src',a.get('href')))
    def handle_data(self,data):
        if self.original_point is not None:self.originals[self.original_point]+=data
        if self.original_fragment is not None:self.original_fragments[self.original_fragment]+=data
        if self.suppressed_tag is None:self.presentation.append(data)
    def handle_endtag(self,tag):
        if tag=='pre':self.original_point=None;self.original_fragment=None
        if tag==self.suppressed_tag:self.suppressed_tag=None


def check_examples(example_ids=None):
    manifest=json.loads((TASK/'compiler_examples_manifest.json').read_text())
    parsers={};file_checks=set();spans=0;links=0
    assert len(manifest['examples'])==4
    workbench=TASK/'link_htmls/max_mlir_mojo_compiler_and_runtime.html'
    home=Parser();home.feed(workbench.read_text())
    parsers[workbench.resolve()]=home
    selected=[e for e in manifest['examples'] if example_ids is None or e['example']['id'] in example_ids]
    for e in selected:
        if e.get('layout') == 'attention-stage-reader-v1':
            from check_attention_walkthrough import check_attention
            report = check_attention()
            assert report['revision'] == manifest['revision']
            file_checks.update(f['path'] for f in e['files'])
            spans += len(e['references'])
            links += report['local_links']
            continue
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
        text_view='text' in payload['toolchain'].get('reading_views',['animation','text'])
        if text_view:
            assert p.original_points==list('1234567'),page
            assert p.originals=={n:answer for n,answer in originals.items() if n!='1'},page
            assert len(p.original_fragments)==3,page
            assert p.diagrams==['build','macro','ir','kernel','offload'],page
        else:
            assert p.original_points==[] and p.original_fragments==[] and p.diagrams==[]
            assert 'toolchain-text' not in p.ids and 'toolchain-text-tab' not in p.ids
            assert 'toolchain-callstack' in p.ids and 'toolchain-animation' in p.ids
        presentation='\n'.join(p.presentation)
        flat_presentation=re.sub(r'\s+','',presentation)
        assert "This Mermaid diagram exceeds the terminal renderer's size limits" not in presentation,page
        assert not re.search(r'(?m)^\s*flowchart (?:LR|TD)\b',presentation),page
        assert 'language-mermaid' not in text,page
        assert payload['toolchain']['originals']==originals,page
        assert payload['toolchain']['point1_sha256']==hashlib.sha256(point1.encode()).hexdigest()==e['toolchain']['point1_sha256']
        if e['example']['id']!='attention':
            assert all(original in payload['toolchain']['text'] for original in originals.values())
        if e['example']['id']=='addition':
            calltrees=payload['toolchain']['calltrees']
            assert calltrees==e['toolchain']['calltrees']
            assert calltrees['stages']==['graph','session','compile','init']
            assert calltrees['branches']==30
            expected_cards=[
                'toolchain-call-graph-context','toolchain-call-graph-package',
                'toolchain-call-graph-location','toolchain-call-graph-module-setup',
                'toolchain-call-graph-op-builder','toolchain-call-graph-type-conversion',
                'toolchain-call-graph-native-builder',
                'toolchain-call-graph-symbolic-params',
                'toolchain-call-graph-retain-body',
                'toolchain-call-graph-device-info',
                'toolchain-call-graph-tracking',
                'toolchain-call-graph-ready-chain',
                'toolchain-call-graph-trailing-chain',
                'toolchain-call-graph-forward-context',
                'toolchain-call-graph-forward-inputs',
                'toolchain-call-graph-forward-callback',
                'toolchain-call-graph-custom-attributes',
                'toolchain-call-graph-custom-insert',
                'toolchain-call-graph-custom-verify',
                'toolchain-call-graph-custom-result',
                'toolchain-call-graph-forward-output',
                'toolchain-call-graph-objects','toolchain-call-graph-validation',
                'toolchain-call-session-ownership','toolchain-call-compile-passes',
                'toolchain-call-compile-kernels','toolchain-call-compile-mef',
                'toolchain-call-init-resources','toolchain-call-init-gates',
                'toolchain-call-example',
            ]
            assert calltrees['cards']==expected_cards
            assert all(card in p.ids for card in expected_cards)
            assert calltrees['reading_order']==[
                'kernels/*.mojo → P1 (.mojoc)','P1 → recorded G1 (MO)',
                'devices → S1','S1 + G1 → clone C1: MO → MAP → MOGG → MGP → CM1',
                'S1 + CM1 → initialized M1',
            ]
            assert 'preparepackageP1' in flat_presentation
            assert 'MO→MAP→MOGG→MGP' in flat_presentation
            assert 'model=session.init(compiled)' in flat_presentation
            tree_text=payload['toolchain']['text'].split('\n\n---\n\n',1)[0]
            assert hashlib.sha256(tree_text.encode()).hexdigest()==calltrees['text_sha256']
            assert len(p.caller_trees)==calltrees['caller_links']
            assert Counter(p.caller_trees)==Counter(calltrees['caller_link_counts'])
            assert Counter(p.call_tree_ids)==Counter({key:1 for key in calltrees['caller_link_counts']})
            assert '↗ caller' in presentation and '[CALLER:' not in presentation
            assert all(tree in p.call_tree_ids for tree in calltrees['separate_call_stack_sections'])
            assert 'class="calltree-nav"' not in text
            stage_cards=expected_cards[:-1]
            assert all(f'id="{card}-launch"' not in text for card in stage_cards)
            legacy_text_cards={'toolchain-call-graph-objects','toolchain-call-graph-validation'}
            assert all(f'data-call-explanation="{card}"' in text for card in stage_cards if card not in legacy_text_cards)
            assert 'id="toolchain-call-example-launch"' in text
            assert 'toolchain-reference' not in p.ids
            assert set(calltrees['references'])<=set(payload['references'])
            assert '[SOURCE:' not in presentation
            assert 'MPKG V2' in presentation and 'waitForModelSetup' in presentation
        elif e['example']['id']=='attention':
            calltrees=payload['toolchain']['calltrees']
            assert calltrees==e['toolchain']['calltrees']
            assert calltrees['stages']==['graph','session','compile','init','execute','kernel']
            assert len(calltrees['cards'])==calltrees['branches']==34
            assert all(card in p.ids for card in calltrees['cards'])
            assert len(p.caller_trees)==calltrees['caller_links']
            assert Counter(p.caller_trees)==Counter(calltrees['caller_link_counts'])
            assert all(tree in p.call_tree_ids for tree in calltrees['separate_call_stack_sections'])
            assert 'modular_ops::fused_attention_custom' in presentation
            assert len(p.attention_scenarios)==len(calltrees['cards'])
            for key,scenarios in p.attention_scenarios.items():
                assert 'toolchain-call-'+key+'-tree' in p.call_tree_ids
                for scenario in scenarios.values():
                    for frame in scenario['frames']:
                        for r in frame['refs']:
                            span=payload['references'][r['key']]
                            assert span['start']<=r['line']<=span['end']
            native=p.attention_scenarios['graph-native-builder']['actual']['frames'][-1]['values']
            assert native['region']=='R0 created' and native['block']=='B0 created'
            assert native['args'].split(', ')==['%q:T','%k:T','%v:T']
            assert 'output=model.execute(Q,K,V)' in flat_presentation
            assert 'forward=None' in presentation and 'BN_1=8' in presentation
            assert 'toolchain-call-example-launch' in p.ids
        else:
            assert 'calltrees' not in payload['toolchain']
        if text_view:
            assert all(f'toolchain-q{n}' in p.ids for n in range(1,8))
            for anchor in ('toolchain-q3-3-1','toolchain-q3-5-2','toolchain-q4-2-2','toolchain-q5-3-1','toolchain-q5-3-2'):
                assert anchor in p.ids,(page,anchor)
            assert 'id="toolchain-text-tab"' in text and 'data-mode-panel="text"' in text
        assert 'id="toolchain-animation-tab"' in text and 'data-mode-panel="animation"' in text
        assert 'ModuleLoad / ExecuteGeneric' in toolchain['ascii'] and 'LLVM TargetRegistry' in toolchain['ascii']
        assert {'mo_to_map','map_to_mogg','mogg_to_mgp','kgen_elaboration','kgen_llvm_passes','llvm_code_gen','package_builder','device_finalization'}<=set(toolchain['refs'])
        assert all(r in payload['references'] for r in p.refs)
        assert Path(e['page']).name in home.links
        for f in e['files']:
            external=f.get('source_origin')=='llvm-project'
            raw=(TASK/f['snapshot']).read_bytes() if external else (REPO/f['path']).read_bytes()
            if f['path'] not in file_checks:
                if external:
                    llvm_manifest=json.loads((TASK/'references/llvm_capi/manifest.json').read_text())
                    assert f['revision']==llvm_manifest['commit']
                    assert any(item['path']==f['path'] and item['sha256']==f['sha256'] for item in llvm_manifest['files'])
                else:
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
    return dict(result='PASS',revision=manifest['revision'],pages=len(selected),examples=[e['example']['id'] for e in selected],animations=sum(len(e['lessons']) for e in selected),visible_mermaid_source=False,source_files=len(file_checks),source_spans=spans,local_links=links,external_runtime_dependencies=0,evidence='Source/document checks only; no MAX execution')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--examples',nargs='+',choices=['addition','epilogue','matmul','attention'])
    args=parser.parse_args()
    report=check_examples(args.examples)
    name='compiler_example_readers_sources.json' if args.examples else 'compiler_examples_sources.json'
    (TASK/'validation'/name).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
