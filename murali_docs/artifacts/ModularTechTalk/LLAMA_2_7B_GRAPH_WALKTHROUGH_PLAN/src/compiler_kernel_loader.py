"""Build the standalone loader/driver reader from pinned complete source files."""
from pathlib import Path
import hashlib
import html
import json
import re

import markdown

from build import cpp_span
from cells import REVISION
from compiler_toolchain import add_toolchain_sources
from compiler_kernel_loader_lessons import SOURCES, EVENTS, SECTIONS, COMPONENTS, OVERVIEW_TREE
from compiler_kernel_loader_research import LLVM_REVISION, LLVM_SOURCES, RESEARCH_LINKS, add_llvm_sources

TASK = Path(__file__).resolve().parent.parent
REPO = TASK.parents[3]
SRC = TASK / 'src'
PAGE = 'max_compiled_kernels_context_load_and_driver.html'


def highlight(text):
    """Font-only emphasis; joining the token text reproduces the input exactly."""
    pattern = r'"[^"\n]*"|\b(?:0x[0-9a-fA-F]+|P[01]|A[01]|JD1|MU1|CODE1|DM1|DF1|B[12]|O[12]|S0|R0|true|false|\d[\d,]*)\b|\b(?:[A-Za-z_]\w*::)+[A-Za-z_]\w*|\b(?:[A-Za-z_]\w*(?=\())|\b(?:IN|OUT|kernelIndex|kernelNames|dylibName|symbolTable|kernels|BufferRef|ArrayRef|StringRef|CompiledFunc|CompiledKernelsContext|StaticArchiveLayer|JITDylib|ObjectLinkingLayer)\b'
    result, end = [], 0
    for match in re.finditer(pattern, text):
        result.append(html.escape(text[end:match.start()]))
        value = match.group()
        kind = 'value' if value.startswith(('"','0x')) or re.fullmatch(r'(?:P[01]|A[01]|JD1|MU1|CODE1|DM1|DF1|B[12]|O[12]|S0|R0|true|false|\d[\d,]*)', value) else 'symbol'
        result.append(f'<span class="token-{kind}">{html.escape(value)}</span>')
        end = match.end()
    result.append(html.escape(text[end:]))
    return ''.join(result)


def build_kernel_loader():
    refs, files = {}, {}
    add_toolchain_sources(refs, files, REPO, TASK, REVISION,
                          sources=[(k,p,'','file',0,k) for k,p in SOURCES.items()])
    add_llvm_sources(TASK, files)
    all_sources = {**SOURCES, **LLVM_SOURCES}

    def link(key, line, label=None):
        assert key in all_sources and 1 <= line <= len(files[all_sources[key]]['text'].splitlines()), (key,line)
        path = all_sources[key]
        return (f'<a href="../sources/{html.escape(path)}#L{line}" data-source="{key}" data-line="{line}" '
                f'title="Open pinned source {html.escape(path)}:L{line}">{html.escape(label or path.split("/")[-1])} · L{line} ↗</a>')

    def tree(rows):
        lines=[]
        for text,key,line in rows:
            lines.append('<span>'+highlight(text).replace('token-','tree-')+' '+link(key,line,'source')+'</span>')
        return '<pre class="call-tree">'+''.join(lines)+'</pre>'

    def teaching_text(value):
        examples={'IMAGE':'cubin (CUDA) / hsaco (HIP) bytes','MODULE_API':'cuModuleLoadDataEx / hipModuleLoadData',
                  'FUNCTION_API':'cuModuleGetFunction / hipModuleGetFunction','LAUNCH_API':'cuLaunchKernelEx / hipModuleLaunchKernel',
                  'MODULE_TYPE':'CUmodule / hipModule_t','FUNCTION_TYPE':'CUfunction / hipFunction_t',
                  'SECOND_SYMBOL':'shape_entry'}
        return re.sub(r'@([A-Z_]+)@',lambda m:examples[m[1]],value)

    def body(e):
        return (f'<div class="values"><div><b class="pink">IN</b><p>{html.escape(teaching_text(e["inputs"]))}</p></div>'
                f'<div><b class="green">OUT</b><p>{html.escape(teaching_text(e["outputs"]))}</p></div></div>'
                f'<p><b>Why:</b> {html.escape(teaching_text(e["why"]))}</p>'
                +('<p class="evidence">The missing-symbol route requests missing_entry as its second name from the start.</p>' if e['id'] in ['caller','arguments'] else '')
                +tree(e['tree']))

    transcript=[]
    for e in EVENTS:
        transcript.append(f'<details data-transcript-event="{e["id"]}"><summary>{html.escape(e["section"]+" · "+teaching_text(e["title"]))} '
                          f'<span class="route-tag">{html.escape(e["route"])} · {e["evidence"]}</span></summary>'+body(e)+'</details>')
    trees=[]
    for section,title in SECTIONS:
        entries=[e for e in EVENTS if e['section']==section]
        content=''.join(f'<div class="tree-event" id="tree-{e["id"]}"><h3>{html.escape(teaching_text(e["title"]))}</h3><span class="route-tag">{e["route"]} · {e["evidence"]}</span>'+body(e)+'</div>' for e in entries)
        trees.append(f'<details class="tree-section" id="section-{section.replace(".","-")}"'+(' open' if section=='2.1' else '')+f'><summary>{section} · {html.escape(title)}</summary>{content}</details>')
    answer=(SRC/'compiler_kernel_loader_answer.md').read_text()
    answer_html=markdown.markdown(answer, extensions=['fenced_code'])
    file_to_key={v:k for k,v in all_sources.items()}

    def answer_link(match):
        href=html.unescape(match[1])
        assert href.startswith(str(REPO)+'/'), href
        path,line=href[len(str(REPO))+1:].rsplit(':',1)
        key=file_to_key[path]
        # Keep the original visible link label, but open the local Source Explorer.
        return link(key,int(line),html.unescape(re.sub('<[^>]+>','',match[2]))).replace(f' · L{line} ↗','')

    answer_html=re.sub(r'<a href="([^"]+)">(.*?)</a>',answer_link,answer_html)
    answer_html=re.sub(r'(<pre><code[^>]*>)(.*?)(</code></pre>)',lambda m:m[1]+highlight(html.unescape(m[2]))+m[3],answer_html,flags=re.S)
    legend=''.join(f'<div class="legend-row" data-color="{key}"><i></i><div><b>{html.escape(title)}</b><p>{html.escape(detail)}</p>{link(source,line)}</div></div>' for key,title,detail,source,line in COMPONENTS)
    raw=files[SOURCES['load']]['text']
    start,end=cpp_span(raw,'ErrorOrSuccess CompiledKernelsContext::load(ArrayRef<uint8_t> binary,',0)
    definition='\n'.join((f'{i:5}  '+raw.splitlines()[i-1]) if raw.splitlines()[i-1] else f'{i:5}'
                         for i in range(start,end+1))
    research=('<p>Primary documentation informed these cutaways. The complete LLVM files are pinned to '
              +link('llvm-config',5,'the checkout’s LLVM selection')+'.</p>'
              +f'<p class="evidence">LLVM revision <code>{LLVM_REVISION}</code>. SOURCE proves the linking mechanism; '
              'the object sections, helper, absolute-pointer fixup and addresses are teaching examples.</p>'
              +'<ul>'+''.join(f'<li><a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">{html.escape(title)}</a>'
                             f' — {html.escape(decision)}</li>' for title,url,decision in RESEARCH_LINKS)+'</ul>'
              +'<p>Offline linking implementation: '
              +link('llvm-object',24,'object → LinkGraph')+' · '
              +link('llvm-link',35,'prune / allocate / resolve / fixup / finalize')+' · '
              +link('llvm-layout',120,'block byte copy')+' · '
              +link('llvm-x86',449,'Pointer64 equation')+' · '
              +link('llvm-memory',85,'memory protection')+' · '
              +link('llvm-core',546,'symbol readiness')+'</p>')
    payload=dict(revision=REVISION,llvm_revision=LLVM_REVISION,events=EVENTS,sections=SECTIONS,sources=all_sources,files=files,
                 answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),evidence='SOURCE / BOUNDARY; illustrative teaching values; no MAX execution')
    replacements={
        'CSS':(SRC/'compiler_card_windows.css').read_text()+'\n'+(SRC/'compiler_kernel_loader.css').read_text()
              +'\n'+(SRC/'compiler_kernel_loader_transformations.css').read_text(),
        'TRANSCRIPT':''.join(transcript),'ANSWER':answer_html,'TREES':''.join(trees),'LEGEND':legend,
        'OVERVIEW':tree(OVERVIEW_TREE),
        'RESEARCH':research,
        'LOADLINK':link('load',85),'DEFINITION':html.escape(definition),'REV':REVISION,
        'DATA':json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'),
        'MODEL':(SRC/'compiler_kernel_loader_model.js').read_text(),
        'VISUALS':(SRC/'compiler_kernel_loader_visuals.js').read_text(),
        'TRANSFORMS':(SRC/'compiler_kernel_loader_transformations.js').read_text(),
        'WINDOWS':(SRC/'compiler_card_windows.js').read_text(),
        'JS':(SRC/'compiler_kernel_loader.js').read_text(),
    }
    shell=(SRC/'compiler_kernel_loader.html').read_text()
    assert set(re.findall(r'@@([A-Z]+)@@',shell))==set(replacements)
    output=re.sub(r'@@([A-Z]+)@@',lambda m:replacements[m[1]],shell)
    (TASK/'link_htmls'/PAGE).write_text(output)
    manifest=dict(revision=REVISION,page=PAGE,events=len(EVENTS),sections=16,
                  llvm_revision=LLVM_REVISION,transformation_lenses=9,
                  research=[dict(title=t,url=u,decision=d) for t,u,d in RESEARCH_LINKS],
                  preserved_answer_sha256=payload['answer_sha256'],
                  source_files={p:f['sha256'] for p,f in files.items()},
                  load_definition=[start,end],evidence=payload['evidence'])
    (TASK/'compiler_kernel_loader_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built {PAGE}: {len(EVENTS)} events, 16 sections, {len(files)} pinned full source files')


if __name__=='__main__':
    build_kernel_loader()
