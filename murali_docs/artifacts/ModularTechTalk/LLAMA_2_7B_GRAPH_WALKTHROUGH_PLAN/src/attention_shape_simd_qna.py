"""Build the copied explanation of scalar SIMD shape attributes."""
from __future__ import annotations

import hashlib
import html
import json
import re

import markdown

from asyncrt_walkthrough import REVISION, SRC, TASK, write
from attention_shape_ape_qna import ShapeApeEvidence
from attention_virtual_device_count import LINK

PAGE = 'max_example_04_shape_dimension_simd_and_constant_folding.html'
ANSWER = 'attention_shape_simd_qna.md'


def render_answer(answer, evidence, *, markdown_in_html=False,
                  link_code_sources=False):
    """Preserve fenced code and optionally render Markdown in disclosure cells."""
    code_blocks = {}

    def source(match):
        label, absolute, number = match.groups()
        path = evidence.source_path(absolute)
        token = f'{path}:{number}'
        return (f'<a class="source-link" data-source="{html.escape(token, quote=True)}" '
                f'href="../sources/{html.escape(path, quote=True)}#L{number}" '
                f'title="{html.escape(token, quote=True)}">{html.escape(label)}</a>')

    def fence(match):
        indent = match['indent']
        lines = match['body'].rstrip('\n').splitlines()
        code = '\n'.join(line[len(indent):] if line.startswith(indent) else line
                         for line in lines)
        if link_code_sources:
            pieces, cursor = [], 0
            for link in LINK.finditer(code):
                pieces.extend((html.escape(code[cursor:link.start()]), source(link)))
                cursor = link.end()
            pieces.append(html.escape(code[cursor:]))
            escaped_code = ''.join(pieces)
        else:
            escaped_code = html.escape(code)
        marker = f'<div data-copied-code="{len(code_blocks)}"></div>'
        code_blocks[marker] = (
            '<div class="code-scroll"><pre><code class="language-' +
            html.escape(match['language'], quote=True) + '">' +
            escaped_code + '</code></pre></div>')
        return indent + marker

    prepared = re.sub(
        r'^(?P<indent> *)```(?P<language>[\w+-]*)[ \t]*\n'
        r'(?P<body>.*?)^(?P=indent)```[ \t]*$',
        fence, answer, flags=re.MULTILINE | re.DOTALL)

    prepared = LINK.sub(source, prepared)
    extensions = ['tables']
    if markdown_in_html:
        extensions.append('md_in_html')
    rendered = markdown.markdown(prepared, extensions=extensions, tab_length=3)
    for marker, block in code_blocks.items():
        if marker not in rendered:
            raise ValueError('Copied code block was lost during rendering')
        rendered = rendered.replace(marker, block)
    return re.sub(r'(<table>.*?</table>)', r'<div class="table-scroll">\1</div>',
                  rendered, flags=re.DOTALL)


def build():
    answer = (SRC / ANSWER).read_text()
    evidence = ShapeApeEvidence()
    for match in LINK.finditer(answer):
        label, absolute, number = match.groups()
        path = evidence.source_path(absolute)
        line = int(number)
        lines = evidence.file(path)['text'].splitlines()
        if not 1 <= line <= len(lines) or not lines[line - 1].strip():
            raise ValueError(f'Invalid copied source anchor: {path}:{line}')
        evidence.references[f'{path}:{line}'] = dict(
            file=path, line=line, label=label, needle=lines[line - 1].strip())

    values = dict(
        answer=render_answer(answer, evidence),
        revision=REVISION,
        source_archive=''.join(
            '<details><summary>' + html.escape(path) + ' · complete source</summary>'
            '<div class="code-scroll"><pre><code>' + html.escape(item['text']) +
            '</code></pre></div></details>' for path, item in evidence.files.items()),
        payload=json.dumps(dict(revision=REVISION, files=evidence.files,
                                references=evidence.references),
                           ensure_ascii=False).replace('<', '\\u003c'),
        css='\n'.join((SRC / name).read_text() for name in
                      ('compiler_card_windows.css', 'attention_moduleop_create.css')),
        js='\n'.join((SRC / name).read_text() for name in
                     ('compiler_card_windows.js', 'attention_virtual_device_count.js')),
    )
    template = (SRC / 'attention_shape_simd_qna.html').read_text()
    tokens = set(re.findall(r'\{\{([a-z_]+)\}\}', template))
    if tokens != values.keys():
        raise ValueError(f'Unexpected SIMD Q&A template fields: {tokens ^ values.keys()}')
    page = re.sub(r'\{\{([a-z_]+)\}\}', lambda match: values[match[1]], template)
    write(TASK / 'link_htmls' / PAGE, page)
    for path, item in evidence.files.items():
        write(TASK / 'sources' / path, item['text'])
    manifest = dict(
        page='link_htmls/' + PAGE,
        revision=REVISION,
        copied_answer='src/' + ANSWER,
        copied_answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
        source_files=len(evidence.files),
        source_anchors=len(evidence.references),
        references=evidence.references,
        files={path: item['sha256'] for path, item in evidence.files.items()},
        evidence='Copied Q&A with pinned MAX source links; no runtime execution',
    )
    write(TASK / 'attention_shape_simd_qna_manifest.json',
          json.dumps(manifest, indent=2) + '\n')
    print(f'Built {PAGE}: one collapsed Q&A, {len(evidence.references)} source anchors')
    return manifest


if __name__ == '__main__':
    build()
