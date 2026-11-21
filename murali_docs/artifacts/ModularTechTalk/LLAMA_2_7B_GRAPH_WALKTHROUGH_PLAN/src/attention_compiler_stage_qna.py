"""Build seven source-linked C1 compiler-stage Q&A supplements."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re

from asyncrt_walkthrough import REVISION, SRC, TASK, write
from attention_chain_bookkeeping_qna import ChainEvidence
from attention_shape_simd_qna import render_answer
from attention_virtual_device_count import LINK

STAGES = [
    dict(id='mo', title='MO · contracts and canonicalization',
         stem='mo_contracts_and_canonicalization',
         overview='A typed attention contract enters the graph optimizer. Follow verification, folding, shape work and the custom-kernel boundary.'),
    dict(id='mo-to-map', title='MO → MAP · explicit access and compute',
         stem='mo_to_map_access_and_compute',
         overview='One mo.custom becomes an opaque MAP region. Track its three inputs, output destination, resolved declaration and parameter dictionary.'),
    dict(id='map', title='MAP · fusion and access optimization',
         stem='map_fusion_and_access_optimization',
         overview='See what the single attention kernel permits, then follow concrete elementwise fusion, operand deduplication and view-motion examples.'),
    dict(id='map-to-mogg', title='MAP → MOGG · pointers, shapes and calls',
         stem='map_to_mogg_implementation_calls',
         overview='The tensor interface gains a body with four pointers and a context. Follow shape construction, value mapping and the registered call.'),
    dict(id='mogg', title='MOGG · call conformance and outlining',
         stem='mogg_call_conformance_and_outlining',
         overview='Move the routine body into a named helper, preserve its caller, and simplify the code generation operations before runtime lowering.'),
    dict(id='mogg-to-mgp', title='MOGG → MGP · model, buffers and chains',
         stem='mogg_to_mgp_models_buffers_and_chains',
         overview='The compiler creates init and execute regions. Follow tensor storage, argument positions, asynchronous handles and dependency edges.'),
    dict(id='mgp', title='MGP · plan optimization and Mojo emission',
         stem='mgp_optimization_and_mojo_emission',
         overview='Optimize the runtime plan, handle the device-graph gate, and turn helper bodies into a Mojo text resource for the later kernel compiler.'),
]
for stage in STAGES:
    stage['page'] = 'max_example_04_c1_' + stage['stem'] + '.html'
    stage['source'] = 'attention_c1_' + stage['stem'] + '.md'
STAGES_BY_ID = {stage['id']: stage for stage in STAGES}

SOURCES = {
    'entry': 'max/examples/custom_ops/fused_attention.py',
    'kernel': 'max/examples/custom_ops/kernels/fused_attention.mojo',
    'graph': 'max/python/max/graph/graph.py',
    'api': 'max/python/max/engine/api.py',
    'frontend': 'GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp',
    'gc': 'GraphCompiler/lib/GraphCompiler.cpp',
    'mo-ops': 'GraphCompiler/lib/MODialect/MOOps.cpp',
    'mo-td': 'GraphCompiler/include/GraphCompiler/MODialect/MOOps.td',
    'mo-fold': 'GraphCompiler/lib/MODialect/Canonicalize.cpp',
    'mo-utils': 'GraphCompiler/lib/MODialect/MOUtils.cpp',
    'mo-options': 'GraphCompiler/include/GraphCompiler/Passes.h',
    'verify-kernels': 'GraphCompiler/lib/MOGGDialect/VerifyKernels.cpp',
    'rewrite-opaque': 'GraphCompiler/lib/MOGGDialect/Transforms/RewriteOpaqueTypes.cpp',
    'verify-params': 'Mojo/lib/Transforms/VerifyParameters.cpp',
    'rmo-legalize': 'GraphCompiler/lib/RMODialect/Transforms/LegalizeRMOOps.cpp',
    'mo-chains': 'GraphCompiler/lib/MODialect/Transforms/CanonicalizeChainSequences.cpp',
    'mo-constant-fold': 'GraphCompiler/lib/MODialect/Transforms/ConstantFold.cpp',
    'mo-hoist': 'GraphCompiler/lib/MODialect/Transforms/HoistConstantSubgraphsIntoParent.cpp',
    'mo-symbolic': 'GraphCompiler/lib/MODialect/Transforms/SymbolicOptimization.cpp',
    'mo-algebra': 'GraphCompiler/lib/MODialect/Patterns/Algebraic.cpp',
    'mo-merge-shapes': 'GraphCompiler/lib/MODialect/Transforms/MergeDuplicateShapeMaterializations.cpp',
    'mo-fallback': 'GraphCompiler/lib/MODialect/Transforms/AddFallbackShapeFuncs.cpp',
    'mo-layouts': 'GraphCompiler/lib/MODialect/Transforms/InferLayouts.cpp',
    'mo-patterns': 'GraphCompiler/lib/MODialect/Transforms/PatternFusion.cpp',
    'mo-pack-shapes': 'GraphCompiler/lib/MODialect/Transforms/InvokeShapePackingFunctions.cpp',
    'mo-propagate': 'GraphCompiler/lib/MODialect/Transforms/PropagateShapesPass.cpp',
    'mo-devices': 'GraphCompiler/lib/MODialect/Transforms/ValidateDevicesPass.cpp',
    'mo-parallel': 'GraphCompiler/lib/MODialect/Transforms/CreateParallelFromUnbundle.cpp',
    'mo-merge-parallel': 'GraphCompiler/lib/MODialect/Transforms/MergeAdjacentParallels.cpp',
    'mo-mutable': 'GraphCompiler/lib/MODialect/Transforms/ElideRedundantMutableOps.cpp',
    'mo-hoist-params': 'GraphCompiler/lib/MODialect/Transforms/HoistParameterExpressionsPass.cpp',
    'mo-remove': 'GraphCompiler/lib/MODialect/Transforms/RemoveUnusedOpsPass.cpp',
    'custom': 'max/python/max/graph/ops/custom.py',
    'registry': 'GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp',
    'registry-support': 'GraphCompiler/lib/MOGGDialect/Analysis/Support.cpp',
    'operand-inference': 'GraphCompiler/lib/Conversion/OperandInference.cpp',
    'params': 'GraphCompiler/lib/Conversion/ParameterInference.cpp',
    'mo-to-map': 'GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp',
    'map-fuse': 'GraphCompiler/lib/MAPDialect/Transforms/FuseKernels.cpp',
    'map-fold': 'GraphCompiler/lib/MAPDialect/Dialect/MAPCanonicalization.cpp',
    'map-to-mogg': 'GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp',
    'map-state': 'GraphCompiler/lib/Interfaces/MAPToMOGGState.cpp',
    'map-lowering': 'GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp',
    'conform': 'GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp',
    'outline': 'GraphCompiler/lib/MOGGDialect/Transforms/OutlineRoutines.cpp',
    'trace': 'GraphCompiler/lib/MOGGDialect/Transforms/WrapFuncsWithTrace.cpp',
    'rowwise': 'GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp',
    'mogg-ops': 'GraphCompiler/lib/MOGGDialect/MOGGOps.cpp',
    'mogg-td': 'GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGOps.td',
    'mogg-to-mgp': 'GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp',
    'lower-state': 'GraphCompiler/lib/Interfaces/LoweringState.cpp',
    'mgp-ops': 'GraphCompiler/lib/MGPDialect/Dialect/MGPOps.cpp',
    'mgp-types': 'GraphCompiler/lib/MGPDialect/Dialect/MGPTypes.cpp',
    'reuse': 'GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp',
    'invariant': 'GraphCompiler/lib/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp',
    'stage-device': 'GraphCompiler/lib/MGPDialect/Transforms/StageDeviceGraphsPass.cpp',
    'finalize-device': 'GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp',
    'emit': 'GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp',
    'emit-state': 'GraphCompiler/lib/Interfaces/EmitMojoState.cpp',
    'jit': 'GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp',
    'lower-routines': 'GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp',
    'test-mo-map': 'GraphCompiler/test/mo-opt/Conversion/MOToMAP/opaque-kernels.mlir',
    'test-elementwise': 'GraphCompiler/test/mo-opt/Conversion/MOToMAP/elementwise-add-static.mlir',
    'test-map-fusion': 'GraphCompiler/test/mo-opt/MAPDialect/Transforms/ElementwiseFusion/basic_pointwise_fusion.mlir',
    'test-map-dedup': 'GraphCompiler/test/mo-opt/MAPDialect/Canonicalization/kernel_operands_basic.mlir',
    'test-view': 'GraphCompiler/test/mo-opt/MAPDialect/Transforms/LoopInvariantViewMotion/loop-invariant-view-motion-transpose.mlir',
    'test-map-mogg': 'GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/plain-opaque.mlir',
    'test-outline': 'GraphCompiler/test/mo-opt/MOGGDialect/Transforms/infra-outline-routines.mlir',
    'test-mgp': 'GraphCompiler/test/mo-opt/Conversion/MOGGToMGP/map-kernel-region.mlir',
    'test-reuse': 'GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-simple-buffer-reuse.mlir',
    'test-chains': 'GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-simplify-chains.mlir',
    'test-mo-shapes': 'GraphCompiler/test/mo-opt/MODialect/Transforms/propagate-shapes.mlir',
    'test-mo-shape-merge': 'GraphCompiler/test/mo-opt/MODialect/Transforms/merge-duplicate-shape-materializations.mlir',
    'test-mo-fusion': 'GraphCompiler/test/mo-opt/MODialect/Transforms/PatternFusion/matmul-add-pattern-fusion.mlir',
    'test-mo-constant': 'GraphCompiler/test/mo-opt/MODialect/Transforms/constant-fold.mlir',
    'test-mo-symbolic': 'GraphCompiler/test/mo-opt/MODialect/Transforms/SymbolicOptimization/elementwise-folding.mlir',
    'test-mo-device': 'GraphCompiler/test/mo-opt/MODialect/ValidateDevicesPass/f8e5m2-cpu-errors.mlir',
}
TOKEN = re.compile(r'\[\[([a-z-]+):(\d+)\]\]')
EXCERPT = re.compile(r'^\{\{excerpt:([a-z-]+):(\d+):(\d+)\}\}$', re.MULTILINE)
QNA = re.compile(r'^## QnA (\d+) · (.+?)(?: \{#([a-z][a-z0-9-]*)\})?$',
                 re.MULTILINE)


def source_link(key, number, evidence):
    path = SOURCES[key]
    line = int(number)
    lines = evidence.file(path)['text'].splitlines()
    if not 1 <= line <= len(lines) or not lines[line - 1].strip():
        raise ValueError(f'Invalid compiler-stage source anchor: {key}:{line}')
    token = f'{path}:{line}'
    label = f'{Path(path).name}:{line}'
    evidence.references[token] = dict(file=path, line=line, label=label,
                                      needle=lines[line - 1].strip())
    return (f'<a class="source-link" data-source="{html.escape(token, quote=True)}" '
            f'href="../sources/{html.escape(path, quote=True)}#L{line}" '
            f'title="{html.escape(token + " — " + lines[line - 1].strip(), quote=True)}">'
            f'↗ {html.escape(label)}</a>')


def render(markdown_text, evidence):
    def excerpt(match):
        key, start, end = match.groups()
        start, end = int(start), int(end)
        lines = evidence.file(SOURCES[key])['text'].splitlines()
        if not 1 <= start <= end <= len(lines):
            raise ValueError(f'Invalid excerpt: {match[0]}')
        body = '\n'.join(lines[start - 1:end])
        return (f'**SOURCE · unchanged excerpt** [[{key}:{start}]]\n\n'
                f'```cpp\n{body}\n```')
    expanded = EXCERPT.sub(excerpt, markdown_text)
    for match in LINK.finditer(expanded):
        label, absolute, number = match.groups()
        path = evidence.source_path(absolute)
        line = int(number)
        lines = evidence.file(path)['text'].splitlines()
        if not 1 <= line <= len(lines):
            raise ValueError(f'Invalid copied compiler-stage source anchor: {path}:{line}')
        evidence.references[f'{path}:{line}'] = dict(
            file=path, line=line, label=label, needle=lines[line - 1].strip())
    # Resolve after fenced-code escaping, so call-tree links work inside <pre>.
    rendered = render_answer(expanded, evidence, markdown_in_html=True,
                             link_code_sources=True)
    return TOKEN.sub(lambda m: source_link(m[1], m[2], evidence), rendered)


def build(stage_ids=None):
    selected = None if stage_ids is None else set(stage_ids)
    if selected is not None and selected - STAGES_BY_ID.keys():
        raise ValueError('Unknown compiler-stage selection')
    manifests = []
    # Share revision checks across pages; each page embeds only its own sources.
    file_cache = {}
    template = (SRC / 'attention_compiler_stage_qna.html').read_text()
    for index, stage in enumerate(STAGES):
        if selected is not None and stage['id'] not in selected:
            continue
        evidence = ChainEvidence()
        evidence.files = file_cache
        answer = (SRC / stage['source']).read_text()
        matches = list(QNA.finditer(answer))
        if not matches or answer[:matches[0].start()].strip():
            raise ValueError(f'Expected numbered Q&A sections in {stage["source"]}')
        cells, qnas = [], []
        for n, match in enumerate(matches):
            number, question, fragment = match.groups()
            if int(number) != n + 1:
                raise ValueError('Non-sequential Q&A number: ' + stage['source'])
            body = answer[match.end():matches[n + 1].start() if n + 1 < len(matches) else len(answer)].strip()
            fragment = fragment or f'qna-{stage["id"]}-{number}'
            if any(qna['id'] == fragment for qna in qnas):
                raise ValueError('Duplicate Q&A fragment: ' + fragment)
            qnas.append(dict(id=fragment, question=question))
            cells.append(f'<details class="qna-cell" id="{fragment}">'
                         f'<summary>QnA {number} · {html.escape(question)}</summary>'
                         '<div class="qna-answer">' + render(body, evidence) + '</div></details>')
        used_files = {ref['file'] for ref in evidence.references.values()}
        page_files = {path: file_cache[path] for path in sorted(used_files)}
        navigation = []
        if index:
            previous = STAGES[index - 1]
            navigation.append(f'<a href="{previous["page"]}">← {html.escape(previous["title"])}</a>')
        if index + 1 < len(STAGES):
            following = STAGES[index + 1]
            navigation.append(f'<a href="{following["page"]}">{html.escape(following["title"])} →</a>')
        values = dict(
            title=html.escape(stage['title']), overview=html.escape(stage['overview']),
            number=str(index + 1), cells='\n'.join(cells), revision=REVISION,
            navigation=''.join(navigation), first_qna=qnas[0]['id'],
            source_archive=''.join(
                '<details><summary>' + html.escape(path) + ' · complete source</summary>'
                '<div class="code-scroll"><pre><code>' + html.escape(item['text']) +
                '</code></pre></div></details>' for path, item in page_files.items()),
            payload=json.dumps(dict(revision=REVISION, files=page_files,
                                    references=evidence.references), ensure_ascii=False).replace('<', '\\u003c'),
            css='\n'.join((SRC / name).read_text() for name in
                          ('compiler_card_windows.css', 'attention_moduleop_create.css')),
            js='\n'.join((SRC / name).read_text() for name in
                         ('compiler_card_windows.js', 'attention_virtual_device_count.js')),
        )
        fields = set(re.findall(r'\{\{([a-z_]+)\}\}', template))
        if fields != values.keys():
            raise ValueError(f'Unexpected compiler-stage template fields: {fields ^ values.keys()}')
        page = re.sub(r'\{\{([a-z_]+)\}\}', lambda m: values[m[1]], template)
        if TOKEN.search(page):
            raise ValueError('Unresolved compiler-stage source token')
        write(TASK / 'link_htmls' / stage['page'], page)
        for path, item in page_files.items():
            write(TASK / 'sources' / path, item['text'])
        manifests.append(dict(
            id=stage['id'], page='link_htmls/' + stage['page'],
            source='src/' + stage['source'],
            source_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            qnas=qnas, source_files=len(page_files), source_anchors=len(evidence.references),
            references=evidence.references,
            files={path: item['sha256'] for path, item in page_files.items()},
        ))
        print(f'Built {stage["page"]}: {len(qnas)} Q&As, {len(evidence.references)} source anchors')
    manifest = dict(
        revision=REVISION, pages=manifests,
        evidence='SOURCE; repository TEST expectations inspected, not executed; schematic IR and illustrative arithmetic; no MAX or GPU execution',
        defaults='All Q&A and archive disclosures collapsed without a fragment',
    )
    manifest_path = TASK / 'attention_compiler_stage_qna_manifest.json'
    if selected is not None and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text())
        if previous_manifest['revision'] != REVISION:
            raise ValueError('Cannot merge compiler stages from different revisions')
        pages = {page['id']: page for page in previous_manifest['pages']}
        pages.update({page['id']: page for page in manifests})
        manifest['pages'] = [pages[stage['id']] for stage in STAGES
                             if stage['id'] in pages]
    write(manifest_path, json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', nargs='+', choices=STAGES_BY_ID)
    build(parser.parse_args().stages)
