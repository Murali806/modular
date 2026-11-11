"""Build the source-pinned C.1 detail reader without modifying compiler code."""
from pathlib import Path
import html
import json
from compiler_pipeline_lessons import LESSONS
from compiler_toolchain import add_toolchain_sources
from cells import REVISION

TASK = Path(__file__).resolve().parent.parent
REPO = TASK.parents[3]
PAGE = 'max_graph_compiler_pipeline_examples.html'
SOURCES = {
 'mo-fold-test': 'GraphCompiler/test/mo-opt/MODialect/misc-ops.mlir',
 'mo-folder-test': 'GraphCompiler/test/mo-opt/MODialect/Transforms/test-folder.mlir',
 'mo-algebra-test': 'GraphCompiler/test/mo-opt/MODialect/Transforms/SymbolicOptimization/elementwise-folding.mlir',
 'mo-shape-test': 'GraphCompiler/test/mo-opt/MODialect/Transforms/merge-duplicate-shape-materializations.mlir',
 'mo-shape-merge': 'GraphCompiler/lib/MODialect/Transforms/MergeDuplicateShapeMaterializations.cpp',
 'mo-unused': 'GraphCompiler/lib/MODialect/Transforms/RemoveUnusedOpsPass.cpp',
 'gc': 'GraphCompiler/lib/GraphCompiler.cpp',
 'opaque': 'GraphCompiler/test/mo-opt/Conversion/MOToMAP/opaque-kernels.mlir',
 'elementwise': 'GraphCompiler/test/mo-opt/Conversion/MOToMAP/elementwise-add-static.mlir',
 'fuse': 'GraphCompiler/lib/MAPDialect/Transforms/FuseKernels.cpp',
 'state': 'GraphCompiler/include/GraphCompiler/Interfaces/MAPToMOGGState.h',
 'state_impl': 'GraphCompiler/lib/Interfaces/MAPToMOGGState.cpp',
 'registry': 'GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp',
 'map-convert': 'GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp',
 'map-lower': 'GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp',
 'map-motion': 'GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp',
 'mgp-convert': 'GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp',
 'device-final': 'GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp',
 'emit': 'GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp',
 'jit_entry': 'GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp',
 'lower': 'GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp',
 'outline': 'GraphCompiler/lib/MGPDialect/Transforms/OutlineModelsToCallables.cpp',
 'frontend': 'GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp',
 'design': 'GraphCompiler/docs/GraphCompiler.md',
 'kernel': 'max/examples/custom_ops/kernels/add_one.mojo',
 'cross': 'GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir',
}
EXTRA = {
 'route': [('frontend',377),('frontend',356)],
 'mo': [('registry',221),('gc',219),('gc',240),('mo-fold-test',649),('mo-folder-test',1),('mo-algebra-test',1),('mo-shape-test',1),('mo-shape-merge',77),('mo-unused',42)],
 'map': [('elementwise',1),('opaque',38),('opaque',92)],
 'map-opt': [('gc',338),('map-motion',1)],
 'mogg': [('registry',48),('registry',167),('map-convert',53),('map-lower',907),('state_impl',1)],
 'mogg-opt': [('gc',370),('map-lower',907)],
 'mgp': [('mgp-convert',116),('mgp-convert',411)],
 'mgp-opt': [('gc',382)],
 'device-graphs': [('device-final',375),('gc',405),('gc',470)],
 'jit': [('jit_entry',529),('cross',1)],
 'runtime-calls': [('lower',61),('lower',137)],
 'callables': [('outline',52),('gc',75)],
}


def build_pipeline_detail():
    refs, files = {}, {}
    entries = [(key,file,'','file',0,key) for key,file in SOURCES.items()]
    add_toolchain_sources(refs,files,REPO,TASK,REVISION,sources=entries)
    def link(key,line,label=None):
        assert 1 <= line <= len(files[SOURCES[key]]['text'].splitlines()), (key,line)
        return f'<a href="../sources/{html.escape(SOURCES[key])}#L{line}" data-source="{key}" data-line="{line}">{html.escape(label or key)} · L{line} ↗</a>'
    nav=''.join(f'<button type="button" data-lesson="{i}">{html.escape(l["title"])}</button>' for i,l in enumerate(LESSONS))
    articles=[]
    for i,l in enumerate(LESSONS):
        links=' · '.join(link(k,n) for k,n in [(l['source'],l['line'])]+EXTRA[l['id']])
        articles.append(f'<article id="{l["id"]}" data-detail="{i}"'+(' hidden' if i else '')+'>'
                        f'<h2>{html.escape(l["title"])}</h2><p class="why"><b>Why needed:</b> {html.escape(l["why"])}</p>'
                        f'<pre class="example"><code>{html.escape(l["example"])}</code></pre>'
                        f'<p>{html.escape(l["detail"])}</p><p class="source-links">SOURCE · {links}</p></article>')
    portable=''.join(f'<details><summary>{html.escape(l["title"])}</summary><p><b>Why:</b> {html.escape(l["why"])}</p><pre>{html.escape(l["example"])}</pre><p>{html.escape(l["detail"])}</p>{link(l["source"],l["line"])}</details>' for l in LESSONS)
    payload=json.dumps(dict(revision=REVISION,lessons=LESSONS,sources=SOURCES,files=files),ensure_ascii=False).replace('</','<\\/')
    shell=(TASK/'src/compiler_pipeline_detail.html').read_text()
    output=shell.replace('@@NAV@@',nav).replace('@@ARTICLES@@',''.join(articles)).replace('@@PORTABLE@@',portable).replace('@@DATA@@',payload).replace('@@REV@@',REVISION)
    output=output.replace('@@CSS@@',(TASK/'src/compiler_pipeline_detail.css').read_text()).replace('@@JS@@',(TASK/'src/compiler_pipeline_detail.js').read_text())
    (TASK/'link_htmls'/PAGE).write_text(output)
    manifest=dict(revision=REVISION,page=PAGE,lessons=len(LESSONS),source_files={p:f['sha256'] for p,f in files.items()},evidence='SOURCE; schematic examples, no MAX execution')
    (TASK/'compiler_pipeline_detail_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built {PAGE}: {len(LESSONS)} compiler stages; {len(files)} pinned complete source files')


if __name__=='__main__':
    build_pipeline_detail()
