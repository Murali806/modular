"""Source-grounded validation workbench; moving references, retained IR, real guards."""
import html
import json


def render_kernel_validation(refs):
    def frame(title, record, detail, point, source, line, **kw):
        return dict(title=title, record=record, detail=detail, packet=record or 'CUST1 contract reference',
                    point=point, source=source, line=line, **kw)
    common = [
        frame('CUST1 is already recorded in G1', '', 'Symbolic %x and %r have type float32[5,10]. P1 is prepared; matrix data is absent.', [28, 118], 'custom_api', 127),
        frame('Select the package Analysis', 'paths(P1) → Analysis A1', 'Python keys its Analysis cache by the package paths. A hit reuses A1; a miss verifies with the current Analysis and caches it after success.', [346, 97], 'ct_graph_verify', 315),
        frame('Ensure the declaration registry', 'A1 → declarations → registry R1', 'Native ensureModule loads package declarations when needed; ensureRegistry indexes them by registration string. This does not recompile P1 or insert its body into G1.', [350, 193], 'ct_validation_registry', 180),
        frame('Look up the registration string', 'add_one → matching declarations', 'containsKernel checks the name read from CUST1. The declaration references AddOne; Python class-name spelling is not the lookup key.', [350, 245], 'ct_validation_native', 642),
        frame('Resolve a compatible device declaration', 'device metadata → chosen AddOne decl', 'getResolvedKernel filters candidates by device info, then ranks user-defined and more-specific matches. This is registration selection, not LLVM backend code emission.', [670, 247], 'ct_validation_device', 167),
        frame('Match input operands first', '%x → input x; MO tensor → tensor adaptor', 'The mapper converts the MO input type to a kernel operand description and checks it against the declared InputTensor. Add-one has one input; an extra operand has no matching slot.', [670, 350], 'ct_validation_mapping', 313),
        frame('Match results to output arguments', '%r → output; destination-passing match', 'Results are matched separately to output declarations. Add-one returns one MO result for its OutputTensor argument. ctx is an implicit launch argument, not another Python tensor input. Parameter-dependent type constraints remain later specialization work.', [670, 350], 'ct_validation_mapping', 321),
        frame('Check supplied compile-time parameters', 'parameters={} → no supplied entries', 'The verifier collects struct/function parameters and checks each supplied name and supported attribute type. Add-one supplies none. Full specialization and parameter-dependent constraints remain later compiler work.', [350, 443], 'ct_validation_params', 426),
        frame('Return success to ops.custom', 'verified CUST1 → return [TensorValue(%r)]', 'The native error result is clear; Python may return its symbolic wrappers. G1 retains CUST1 and P1 remains separate. No Mojo arithmetic, lowering or hardware execution has run here.', [30, 470], 'custom_api', 129),
    ]
    for f, packet in zip(common, ['CUST1 contract reference', 'ref(paths P1)', 'ref(declarations)', 'lookup(add_one)', 'ref(selected decl)', 'ref(%x) → input x', 'ref(%r) → output', 'supplied params: {}', 'ref(%r) → wrapper']):
        f['packet'] = packet
    scenarios = {'valid': common}
    scenarios['unknown'] = common[:3] + [frame('Stop: registration is missing', 'Error: no kernel registered for add_one_typo', 'containsKernel fails before device selection or operand mapping. Native Analysis raises ValueError; ops.custom does not return successful result wrappers. The already inserted IR is shown retained, not rolled back.', [350, 245], 'ct_validation_native', 643, failure=True)]
    scenarios['extra'] = common[:5] + [frame('Stop: second input has no matching argument', 'Error: CUST1 has too many operands', 'Teaching variant values=[x,x]: first %x maps to x; the second input has no kernel input. Input matching returns an error before output mapping and parameter checking. Native Analysis propagates ValueError.', [670, 350], 'ct_validation_mapping', 253, failure=True)]
    scenarios['unknown'][-1]['packet'] = 'lookup(add_one_typo) → error'
    scenarios['extra'][-1]['packet'] = 'second input → no slot'
    for frames in scenarios.values():
        for i, f in enumerate(frames):
            f['index'] = i
            f['point'][1] = 552
            f['sourceFile'] = refs[f['source']]['file']
            f['sourceLabel'] = refs[f['source']]['label']
    data = html.escape(json.dumps(common, ensure_ascii=False), quote=True)
    variants = html.escape(json.dumps(scenarios, ensure_ascii=False), quote=True)
    def link(key, line):
        r = refs[key]
        return f'<a data-ref="{key}" data-line="{line}" href="../sources/{html.escape(r["file"])}#L{line}">{html.escape(r["label"])}</a>'
    def box(x, y, w, h, title, lines, cls='conversion-container', extra=''):
        text = f'<text x="{x+12}" y="{y+24}" class="validation-heading">{html.escape(title)}</text>'
        text += ''.join(f'<text x="{x+12}" y="{y+48+i*20}">{html.escape(t)}</text>' for i, t in enumerate(lines))
        return f'<g {extra}><rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>{text}</g>'
    svg = box(14, 16, 306, 526, 'G1 · CALLER-OWNED MO GRAPH', ['CUST1 exists before this verification'], extra='data-validation-owner="G1"')
    svg += box(28, 80, 278, 220, 'CUST1 · mo.custom', ['', 'input %x : float32[5,10]', 'result %r : float32[5,10]', 'device = selected CPU / GPU', 'parameters = {}'], 'conversion-source')
    svg += '<text x="40" y="128" data-validation-symbol>symbol = add_one</text>'
    svg += '<g data-validation-extra opacity="0"><rect x="38" y="253" width="253" height="30" rx="4" class="validation-error"/><text x="45" y="273">extra input %x · no matching slot</text></g>'
    svg += box(28, 334, 278, 90, 'P1 · PREPARED .mojoc', ['AddOne.execute remains in P1', 'No body inlining / no execution'], 'conversion-source')
    svg += box(28, 455, 278, 69, 'RESULT WRAPPERS', ['Not returned yet'], extra='data-validation-result')
    svg += box(334, 16, 608, 526, 'NATIVE COMPILER ANALYSIS · SAME PROCESS', ['References and declarations; no tensor data moves'])
    svg += box(346, 80, 576, 77, 'A1 · package Analysis', ['Package-path cache / lazy declaration module'], extra='data-validation-phase="1" opacity="0"')
    svg += box(346, 176, 290, 225, 'R1 · declaration index', ['registration → declarations', 'add_one → AddOne', 'other names → other declarations'], extra='data-validation-phase="2" opacity="0"')
    svg += '<rect x="354" y="277" width="273" height="33" rx="4" data-validation-gate="name"/><text x="364" y="299" data-validation-label="name">Name lookup pending</text>'
    svg += box(658, 176, 264, 225, 'SELECTED DECLARATION', ['execute[target](output,x,ctx)', 'x: InputTensor', 'output: OutputTensor', 'ctx: implicit launch arg'], extra='data-validation-phase="4" opacity="0"')
    svg += '<rect x="666" y="359" width="246" height="29" rx="4" data-validation-gate="device"/><text x="676" y="378" data-validation-label="device">Device selection pending</text>'
    svg += '<g data-validation-input-map opacity="0"><path d="M290 149 H325 V334 H662" class="validation-match"/><text x="352" y="328">%x → input x · match</text></g>'
    svg += '<g data-validation-output-map opacity="0"><path d="M290 169 H316 V362 H662" class="validation-match"/><text x="352" y="355">%r → output · match</text></g>'
    svg += '<g data-validation-rejected-input opacity="0"><path d="M290 267 H321 V400 H654" class="validation-reject"/><text x="351" y="394">second %x → NO SLOT ×</text></g>'
    svg += '<rect x="346" y="420" width="576" height="42" rx="5" data-validation-gate="params"/><text x="359" y="447" data-validation-label="params">Parameter check pending</text>'
    svg += '<rect x="346" y="479" width="576" height="44" rx="5" data-validation-gate="outcome"/><text x="359" y="506" data-validation-label="outcome">Validation pending · wrappers not returned</text>'
    svg += '<path class="conversion-route" d="M295 115 H334 M295 369 H327 V198 H345 M635 293 H657"/>'
    svg += '<g data-conversion-packet transform="translate(28,552)"><path d="M0 0 H170 L182 12 V45 H0 Z"/><foreignObject x="8" y="5" width="161" height="36"><div xmlns="http://www.w3.org/1999/xhtml" data-conversion-packet-label>CUST1 contract reference</div></foreignObject></g>'
    svg += box(14, 615, 928, 286, 'VALIDATION HISTORY · each guard leaves evidence', [])
    svg += ''.join(f'<g data-conversion-record="{i}" opacity="0"><rect x="27" y="{648+i*28}" width="901" height="24" rx="3"/><text x="39" y="{665+i*28}" data-validation-record-label="{i}"></text></g>' for i in range(1, 9))
    transcript = ''.join(f'<li><b>{html.escape(f["title"])}</b> → <code>{html.escape(f["record"])}</code>. {html.escape(f["detail"])} {link(f["source"], f["line"])}</li>' for f in common[1:])
    sources = ' · '.join(link(k, n) for k, n in [('ct_analysis_verify', 69), ('ct_validation_lookup', 48), ('ct_validation_native', 634), ('ct_validation_device', 167), ('ct_validation_mapping', 313), ('ct_validation_params', 426)])
    return (f'<section class="card-conversion kernel-validation-conversion" data-card-conversion="custom-verify" data-conversion-frames="{data}" data-validation-scenarios="{variants}" data-validation-scenario="valid" data-conversion-index="0">'
            '<h3>How CUST1 is validated against P1</h3><p class="conversion-evidence"><b>SOURCE</b> · Schematic declarations and reference handoffs, not a compiler dump or measured execution.</p>'
            '<label class="validation-scenario">Call to inspect <select data-validation-select><option value="valid">Original add-one · valid</option><option value="unknown">Teaching variant · unknown name</option><option value="extra">Teaching variant · extra input</option></select></label>'
            '<div class="conversion-controls"><button type="button" data-conversion-action="back">Back</button><button type="button" data-conversion-action="step">Step</button><button type="button" data-conversion-action="play">Play</button><button type="button" data-conversion-action="reset">Reset</button><output data-conversion-count>0 / 8</output></div>'
            f'<div class="conversion-canvas"><svg viewBox="0 0 960 918" role="img" aria-label="Retained MO call and Mojo package; registration lookup, device resolution, operand matching, parameter guard and return">{svg}</svg></div>'
            '<p class="conversion-current" aria-live="polite"><b data-conversion-title>CUST1 is already recorded</b><span data-conversion-detail>Use Step or Play to follow the references.</span></p>'
            '<p class="validation-current-source">Current guard: '+link('custom_api', 127).replace('<a ', '<a data-validation-source ' ,1)+'</p>'
            f'<details class="conversion-transcript"><summary>Portable validation flow · every guard · source evidence</summary><ol>{transcript}</ol><p>Unknown name stops at containsKernel; extra input stops during input mapping. Both propagate via native Analysis as ValueError. G1 retains the inserted call; no rollback is illustrated.</p><p>{sources}</p></details></section>')
