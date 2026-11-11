"""Pointer bridge animation: multiple wrappers, one attached native GraphOp."""
import html
import json


def render_capi_conversion(refs, *, graph_name="addition", inputs="%x"):
    native_label = 'OP1: mo.graph @addition at SAME P' if graph_name == 'addition' else f'OP1: @{graph_name} at SAME P'
    body_label = 'region R0 → block B0 → input %x' if inputs == '%x' else f'R0 → B0 → {inputs}'
    frames = [
        dict(title='Native GraphOp already exists', record='', detail='MAX op refers to OP1 inside MODULE1. Native GraphOp count stays 1 throughout.', packet='OP1 at symbolic address P', index=0),
        dict(title='Export _CAPIPtr as a Python capsule', record='CAP1: named PyCapsule, pointer=P', detail='MAX calls wrap(getOperation()), then PyCapsule_New. No IR data is serialized; the capsule has no keep-alive destructor.', packet='CAP1 carries the same pointer P', index=1),
        dict(title='Extract and check the C API handle', record='MlirOperation { ptr=P }', detail='PyCapsule_GetPointer checks the expected operation capsule name. The valid path retains P; a wrong name raises.', packet='C handle: MlirOperation{P}', index=2),
        dict(title='Create the MLIR Python wrapper', record='G1._mlir_op → PyOperation → OP1', detail='The factory obtains OP1’s context and creates a PyOperation wrapper. OP1 remains attached to MODULE1.', packet='PyOperation holds P + context ref', index=3),
        dict(title='Read the existing region and body', record='_current_block = _graph_body = B0', detail='regions[0].blocks[0] retrieves B0. No GraphOp, region or block is cloned or created by this lookup.', packet='Existing R0 / B0 handles', index=4),
    ]
    data = html.escape(json.dumps(frames, ensure_ascii=False), quote=True)
    records=''.join(f'<g data-conversion-record="{i}" opacity="0"><rect x="334" y="{70+i*33}" width="350" height="28" rx="4"/><text x="342" y="{89+i*33}">{html.escape(f["record"])}</text></g>' for i,f in enumerate(frames[1:]))
    r=refs['ct_capi_export']
    def source(key):
        r=refs[key]
        return f'<a data-ref="{key}" data-line="{r["line"]}" href="../sources/{html.escape(r["file"])}#L{r["line"]}">{html.escape(r["label"])}</a>'
    steps=''.join(f'<li><b>{html.escape(f["title"])}</b> → <code>{html.escape(f["record"])}</code>. {html.escape(f["detail"])}</li>' for f in frames[1:])
    return (f'<section class="card-conversion capi-conversion" data-card-conversion="retain-body" data-conversion-frames="{data}" data-conversion-index="0" aria-label="C API pointer bridge: one native GraphOp">'
        '<h3>MAX wrapper → C API capsule → MLIR wrapper</h3>'
        '<p class="conversion-evidence"><b>SOURCE</b> · Same-process pointer bridge. P is a symbolic address; no measured addresses or runtime execution.</p>'
        '<div class="conversion-controls"><button type="button" data-conversion-action="back">Back</button><button type="button" data-conversion-action="step">Step</button><button type="button" data-conversion-action="play">Play</button><button type="button" data-conversion-action="reset">Reset</button><output data-conversion-count>0 / 4</output></div>'
        '<div class="conversion-canvas"><svg viewBox="0 0 720 450" role="img" aria-label="MAX and MLIR Python wrappers reference one native operation via a named capsule">'
        '<rect class="conversion-source" x="14" y="20" width="275" height="65" rx="8"/>'
        '<text x="25" y="43">MAX Python _core wrapper: op</text><text x="25" y="65">getOperation() → OP1 at P</text>'
        '<g data-capi-phase="1" opacity="0"><rect x="14" y="100" width="275" height="52" rx="6" fill="#fff0ca" stroke="#8c5b00"/><text x="25" y="121">CAP1: PyCapsule</text><text x="25" y="140">operation name · P · destructor=NULL</text></g>'
        '<g data-capi-phase="3" opacity="0"><rect x="14" y="175" width="275" height="58" rx="6" fill="#ede7ff" stroke="#6840a0"/><text x="25" y="197">MLIR Python PyOperation wrapper</text><text x="25" y="218">same P + context reference</text></g>'
        '<rect class="conversion-container" x="322" y="20" width="375" height="270" rx="8"/><text x="336" y="43">BRIDGE RECORDS · no native clone</text>' + records +
        '<path class="conversion-route" d="M150 85 V100 M150 152 V175 M150 233 V320"/>'
        '<rect class="conversion-container" x="14" y="320" width="683" height="116" rx="8"/>'
        '<text x="28" y="343">MODULE1 · native builtin.module owns its attached GraphOp</text>'
        f'<g data-capi-native="OP1"><rect x="28" y="355" width="368" height="65" rx="6" fill="#e7f4ec" stroke="#2c6e5a"/><text x="40" y="378">{html.escape(native_label)}</text><text x="40" y="400">{html.escape(body_label)}</text></g>'
        '<text x="416" y="378">Native GraphOp count = 1</text><text x="416" y="401">IR remains in MODULE1</text>'
        '<g data-capi-phase="3" opacity="0"><path d="M50 233 V355" fill="none" stroke="#6840a0" stroke-width="2"/><text x="62" y="306">wrapper reference to P</text></g>'
        '<g data-conversion-packet transform="translate(22,110)"><path d="M0 0 H244 L258 14 V60 H0 Z"/><text x="10" y="22">POINTER / REFERENCE HANDOFF</text><foreignObject x="9" y="28" width="240" height="30"><div xmlns="http://www.w3.org/1999/xhtml" data-conversion-packet-label>OP1 at symbolic address P</div></foreignObject></g>'
        '</svg></div><p class="conversion-current" aria-live="polite"><b data-conversion-title>Native GraphOp already exists</b><span data-conversion-detail>Native GraphOp count stays 1.</span></p>'
        '<p class="conversion-capi-sources">'+source('ct_capi_export')+' · '+source('capi_capsule')+' · '+source('capi_factory')+'</p>'
        '<details class="conversion-transcript"><summary>Portable pointer handoff</summary><ol>'+steps+'</ol></details></section>')
