"""Embed every lesson and its complete source-linked transcript for offline use."""
from copy import deepcopy
import html
import json


def resolve_scenarios(item, refs):
    scenarios = deepcopy(item['scenarios'])
    for scenario in scenarios.values():
        for frame in scenario['frames']:
            resolved=[]
            for key,line in frame['refs']:
                r=refs[key]
                if not r['start'] <= line <= r['end']:
                    raise ValueError(f"Attention lesson {item['key']} source outside span: {key}:{line}")
                resolved.append(dict(key=key,line=line,file=r['file'],label=r['label']))
            frame['refs']=resolved
    return scenarios


def links(frame):
    return ' · '.join(f'<a data-ref="{r["key"]}" data-line="{r["line"]}" '
                      f'href="../sources/{html.escape(r["file"])}#L{r["line"]}">'
                      f'{html.escape(r["label"])} · L{r["line"]}</a>' for r in frame['refs'])


def render_attention_mechanism(item, refs, src):
    scenarios=resolve_scenarios(item,refs)
    first=scenarios['actual']['frames'][0]
    esc=lambda v: html.escape(str(v),quote=True)
    data=esc(json.dumps(scenarios,ensure_ascii=False))
    selector=''.join(f'<option value="{esc(k)}">{esc(s["label"])}</option>' for k,s in scenarios.items())
    controls=('<div class="attention-motion-controls" data-interactive hidden>'
              '<button type="button" data-am-action="back">Back</button>'
              '<button type="button" data-am-action="step">Step</button>'
              '<button type="button" data-am-action="play">Play</button>'
              '<button type="button" data-am-action="reset">Reset</button>'
              '<output data-am-count>1</output>'
              f'<label class="attention-case">Case <select data-am-scenario>{selector}</select></label>'
              '<label>Step time <select data-am-speed><option value="3200">3.2 s</option>'
              '<option value="1600">1.6 s</option><option value="6000">6 s</option></select></label>'
              '<label class="attention-seek">Timeline <input data-am-slider type="range" min="0" step="1" '
              f'max="{len(scenarios["actual"]["frames"])-1}" value="0"></label></div>')
    static=''.join(f'<div class="attention-object"><span class="attention-object-kind">{esc(n["kind"])} · {esc(n["owner"])}</span>'
                   f'<b>{esc(n["label"])}</b><pre>{esc(first["values"].get(n["id"]) if first["values"].get(n["id"]) is not None else "not created / not reached")}</pre></div>'
                   for n in scenarios['actual']['nodes'])
    transcript=[]
    for key,scenario in scenarios.items():
        steps=[]
        for i,frame in enumerate(scenario['frames']):
            steps.append(f'<li><button type="button" data-am-seek="{i}" data-am-case="{esc(key)}">Step {i+1}</button> '
                         f'<b>{esc(frame["title"])}</b><dl><dt>IN</dt><dd>{esc(frame["input"])}</dd>'
                         f'<dt>DO</dt><dd>{esc(frame["action"])}</dd><dt>OUT</dt><dd>{esc(frame["output"])}</dd>'
                         f'<dt>WHY</dt><dd>{esc(frame["why"])}</dd></dl><div>{links(frame)}</div></li>')
        transcript.append(f'<details class="attention-case-transcript"'+(' open' if key=='actual' else '')+'>'
                          f'<summary>{esc(scenario["label"])}</summary><p>{esc(scenario["note"])}</p><ol>'
                          +''.join(steps)+'</ol></details>')
    return (f'<section class="attention-mechanism" data-attention-mechanism="{esc(item["view"])}" '
            f'data-am-lesson="{esc(item["key"])}" data-am-scenarios="{data}" data-am-index="0" data-am-case="actual" '
            f'aria-label="{esc(item["question"])}">'
            '<h3>Mechanism · step by step</h3>'
            f'<p class="attention-question">{esc(item["question"])}</p>'
            f'<div class="attention-io-contract"><p><b>IN</b> {esc(item["inputs"])}</p><p><b>OUT</b> {esc(item["outputs"])}</p></div>'
            '<p class="attention-motion-evidence"><b>SOURCE</b> · Pinned implementation; illustrative identities/IR. '
            'Numerical scenes run in the browser, not MAX/GPU. Driver/hardware timing is a <b>BOUNDARY</b>.</p>'
            +controls+'<p class="attention-scenario-note" data-am-note></p>'
            f'<div class="attention-motion-event" aria-live="polite"><b data-am-title>{esc(first["title"])}</b>'
            '<dl><dt>IN</dt><dd data-am-step-input>'+esc(first['input'])+'</dd><dt>DO</dt><dd data-am-step-action>'+esc(first['action'])
            +'</dd><dt>OUT</dt><dd data-am-step-output>'+esc(first['output'])+'</dd><dt>WHY</dt><dd data-am-step-why>'+esc(first['why'])+'</dd></dl></div>'
            '<div class="attention-motion-sources" data-am-sources>'+links(first)+'</div>'
            f'<div class="attention-canvas" data-am-canvas><div class="attention-objects">{static}</div></div>'
            '<details class="attention-motion-transcript"><summary>Portable flow · step-by-step transcript · evidence</summary>'
            +''.join(transcript)+'</details></section>')
