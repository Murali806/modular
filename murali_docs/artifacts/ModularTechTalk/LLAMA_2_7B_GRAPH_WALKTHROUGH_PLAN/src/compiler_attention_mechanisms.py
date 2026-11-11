"""Render attention card mechanisms from the same model used in the browser."""
import html
import json
import subprocess

TITLES = {
    "package": "Mojo source package → checked IR → P1 bytecode",
    "context": "Current thread → active MLIR context", "location": "Python call-site → IR Location",
    "module": "Selected module → graph container", "builder": "Insertion point → new operation",
    "type": "Python type fields → native MO tensor type", "graph": "Signature → region → block arguments",
    "inputs": "Native arguments → three symbolic wrappers", "custom": "Python custom call → MO fields",
    "validation": "Registry lookup → operand mapping → verified call", "output": "Symbolic result → complete G1",
    "compiler": "Retained G1 → changing compiler clone C1", "kernels": "Mojo specialization → LLVM → target artifacts",
    "session": "Device references → configs → runtime ownership", "init": "Prepared bytes → loaded handles → M1",
    "execute": "Three buffers → native call → one output", "launch": "Specialized function → grid → output tiles",
    "mma": "Operand fragments → MMA → shared tile → barrier", "storage": "Current tiles and surviving state",
    "online": "Two key tiles → stable online softmax → output",
}


def mechanism_frames(kind, refs, src):
    script = "process.stdout.write(JSON.stringify(require('./compiler_attention_motion.js').frames(process.argv[1])))"
    frames = json.loads(subprocess.check_output(["node", "-e", script, kind], cwd=src))
    for f in frames:
        resolved = []
        for item in f["refs"]:
            key, line = item[0], item[1] if len(item) > 1 else refs[item[0]]["line"]
            r = refs[key]
            if not r["start"] <= line <= r["end"]:
                raise ValueError(f"Attention animation source outside span: {key}:{line}")
            resolved.append(dict(key=key, line=line, file=r["file"], label=r["label"]))
        f["refs"] = resolved
    return frames


def render_attention_mechanism(kind, refs, src):
    frames = mechanism_frames(kind, refs, src)
    title = TITLES[kind]
    data = html.escape(json.dumps(frames, ensure_ascii=False), quote=True)
    controls = ('<div class="attention-motion-controls" data-interactive hidden><button type="button" data-am-action="back">Back</button>'
                '<button type="button" data-am-action="step">Step</button><button type="button" data-am-action="play">Play</button>'
                '<button type="button" data-am-action="reset">Reset</button><output data-am-count>0</output></div>')
    if kind == "online":
        rows = ''.join(f'<tr data-am-key="{i}"><th>K{i} / V{i}</th><td data-am-score>{frames[0]["scores"][i]}</td>'
                       f'<td data-am-exp>—</td><td data-am-weight>—</td><td data-am-value>{frames[0]["v"][i]}</td>'
                       '<td data-am-contribution>—</td></tr>' for i in range(4))
        canvas = ('''<div class="attention-numeric-scene">
<div class="attention-state-values"><div><span>running max m</span><b data-am-m>−∞</b></div><div><span>denominator l</span><b data-am-l>0</b></div><div><span>retained output O</span><b data-am-o>[0,0]</b></div></div>
<p class="attention-tile-label" data-am-tile>Inputs ready · Q row [1,0]</p>
<div class="attention-table-scroll" tabindex="0"><table class="attention-numeric-table"><thead><tr><th>row</th><th>Q·K</th><th>exp(S−m)</th><th>E/l</th><th>V row</th><th>weighted V</th></tr></thead><tbody>''' + rows + '''</tbody></table></div>
<div class="attention-denominator"><span>Denominator = prior mass + current tile mass</span><div class="attention-mass-bar"><div data-am-old-mass></div><div data-am-new-mass></div></div><p data-am-mass-text>Old mass=0; new tile not visited</p></div>
<div class="attention-output-equation"><div><span>Rescaled prior O</span><b data-am-old-o>[0,0]</b></div><strong>+</strong><div><span>Current tile PV</span><b data-am-new-o>[0,0]</b></div><strong>=</strong><div><span>New normalized O</span><b data-am-sum-o>[0,0]</b></div></div>
<p class="attention-reference" data-am-reference>Full-softmax reference is checked after both key tiles.</p></div>''')
    else:
        names = list(dict.fromkeys(key for f in frames for key in f.get("fields", {})))
        fields = ''.join(f'<div class="attention-field" data-am-field="{html.escape(key)}"><span>{html.escape(key)}</span><b>{html.escape(frames[0]["fields"].get(key,"pending"))}</b></div>' for key in names)
        canvas = (f'''<div class="attention-transform-scene"><div class="attention-transform-panes"><div><h4>Input / previous form</h4><pre data-am-before>{html.escape(frames[0]["before"])}</pre></div><div><h4>New form / retained result</h4><pre data-am-after>{html.escape(frames[0]["after"])}</pre></div></div>
<div class="attention-motion-route"><span>source / caller</span><span>result / next consumer</span><div class="attention-motion-packet" data-am-packet><b data-am-packet-text>named payload</b></div></div>
<div class="attention-artifact-fields">''' + fields + '</div></div>')
    transcript = []
    for i, f in enumerate(frames):
        sources = ' · '.join(f'<a data-ref="{r["key"]}" data-line="{r["line"]}" href="../sources/{html.escape(r["file"])}#L{r["line"]}">{html.escape(r["label"])} · L{r["line"]}</a>' for r in f["refs"])
        values = (f.get('after', '') if kind != 'online' else
                  f'm={f["m"] if f["m"] is not None else "−∞"}, l={f["l"]:.6f}, O={f["O"]}')
        transcript.append(f'<li><button type="button" data-am-seek="{i}">{i}</button> <b>{html.escape(f["title"])}</b>'
                          f'<pre>{html.escape(values)}</pre><p>{html.escape(f["detail"])}</p><div>{sources}</div></li>')
    return (f'<section class="attention-mechanism" data-attention-mechanism="{kind}" data-am-frames="{data}" data-am-index="0" '
            f'aria-label="{html.escape(title)}"><h3>{html.escape(title)}</h3>'
            '<p class="attention-motion-evidence"><b>SOURCE</b> · Schematic IR, identities and storage. Tiny arithmetic runs in this browser; no MAX/GPU run.</p>'
            + controls + canvas + f'<p class="attention-motion-event" aria-live="polite"><b data-am-title>{html.escape(frames[0]["title"])}</b><span data-am-detail>{html.escape(frames[0]["detail"])}</span></p>'
            '<div class="attention-motion-sources" data-am-sources></div>'
            '<details class="attention-motion-transcript"><summary>Portable flow · step-by-step transcript · evidence</summary><ol>'
            + ''.join(transcript) + '</ol></details></section>')
