"""Build the source-grounded, offline Linux/MAX/GPU systems lab."""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess

from call_stack_refs import FILES as STACK_FILES, REFS as STACK_REFS
from memory_planning import FILES as MEMORY_FILES, REFS as MEMORY_REFS, definition
from serve_story import render_legend, render_stories, story_contract, story_markdown

PAGE_NAME = "max_serve_os_driver_and_scaling.html"
BASE = "max/python/max/"
FILES = {**STACK_FILES, **MEMORY_FILES, **{key: BASE + path for key, path in {
    "PROC": "serve/process_control.py",
    "IPC": "serve/worker_interface/_zmq_queue.py",
    "API_LLM": "serve/pipelines/llm.py",
    "OPENAI": "serve/router/openai_routes.py",
    "TG_SCHED": "serve/scheduler/text_generation_scheduler.py",
    "BATCH": "serve/scheduler/batch_constructor/text_batch_constructor.py",
    "BUFFER": "driver/buffer.py",
    "HOST_MEMORY": "support/host_memory.py",
}.items()}}
REFS = {key: STACK_REFS[key] for key in (
    "cli", "serve", "config_call", "bundle", "app", "uvicorn_cfg", "eventloop",
    "lifespan", "start_worker", "spawn", "ready", "connected", "invoke",
    "pipeline_init", "worker_run", "alive", "http", "load_weights",
)}
REFS.update({key: MEMORY_REFS[key] for key in (
    "load_devices", "load_device", "free_memory", "plan_type", "plan_sizes",
    "manager_load", "bytes_per_block", "geometry",
)})
REFS.update({
    "fd_wait": definition("PROC", "_read_ready"),
    "process_ready": definition("PROC", "ProcessManager.ready"),
    "ipc_address": definition("IPC", "generate_zmq_ipc_path"),
    "ipc_config": definition("IPC", "ZmqConfig.__init__"),
    "proxy_submit": definition("ZMQ", "ZmqModelWorkerProxy.stream"),
    "proxy_init": definition("ZMQ", "ZmqModelWorkerProxy.__init__"),
    "proxy_drain": definition("ZMQ", "ZmqModelWorkerProxy._drain_responses"),
    "proxy_responses": definition("ZMQ", "ZmqModelWorkerProxy.response_worker"),
    "proxy_cancel": definition("ZMQ", "ZmqModelWorkerProxy.cancel"),
    "http_429": definition("API", "_request_queue_full_exception_handler"),
    "queue_settings": definition("SETTINGS", "Settings", "Settings.max_queue_size and max_pending_requests", "    max_queue_size: int | None = Field("),
    "first_chunk": definition("OPENAI", "_start_stream"),
    "sse_response": definition("OPENAI", "openai_create_chat_completion", "Return the SSE response after admission and first chunk", '            return EventSourceResponse(token_stream, ping=100000, sep="\\n")'),
    "tokenizer_cpu": definition("TOKENIZER", "TextTokenizer._encode_text"),
    "tokenizer_executor": definition("TOKENIZER", "run_with_default_executor"),
    "token_context": definition("TOKENIZER", "TextTokenizer.new_context"),
    "token_chunks": definition("API_LLM", "TokenGeneratorPipeline.next_token_chunk", "Tokenize, submit and drain token chunks", "                context = await self.tokenizer.new_context(request)"),
    "scheduler_pending": definition("TG_SCHED", "TokenGenerationScheduler._retrieve_pending_requests"),
    "scheduler_iteration": definition("TG_SCHED", "TokenGenerationScheduler.run_iteration"),
    "scheduler_execute": definition("TG_SCHED", "TokenGenerationScheduler._schedule", "Execute, advance requests and release completed requests", "                responses = self.pipeline.execute(inputs)"),
    "construct_batch": definition("BATCH", "TextBatchConstructor.construct_batch"),
    "release_request": definition("BATCH", "TextBatchConstructor.release_request"),
    "preempt_request": definition("BATCH", "TextBatchConstructor._preempt_request"),
    "pipeline_execute": definition("TEXT", "TextGenerationPipeline._execute"),
    "llama_execute": definition("MODEL", "LlamaModelBase.execute"),
    "session_native": definition("ENGINE", "InferenceSession.__init__", "Python session delegates to the native engine", "        self._impl = _InferenceSession("),
    "buffer_copy": definition("BUFFER", "inplace_copy_from"),
    "buffer_host": definition("BUFFER", "_to_numpy"),
    "host_memory_limit": definition("HOST_MEMORY", "host_memory_limit"),
})
CODE_CELLS = ["spawn", "fd_wait", "process_ready", "ipc_address", "proxy_submit",
              "http_429", "tokenizer_cpu", "scheduler_pending", "llama_execute",
              "buffer_host", "release_request"]


def sequence_svg(scene):
    """Draw one compact actor-complete sequence; JavaScript animates its marker."""
    width = max(900, 40 + 174 * len(scene["actors"]))
    positions = {actor[0]: 105 + index * 174 for index, actor in enumerate(scene["actors"])}
    pieces = [f'<svg viewBox="0 0 {width} 300" role="group" aria-labelledby="title-{scene["id"]}" data-sequence>',
              f'<title id="title-{scene["id"]}">{html.escape(scene["title"])}: actors, ownership and event order</title>',
              f'<defs><marker id="arrow-{scene["id"]}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" class="arrowhead"/></marker></defs>']
    for key, label, subtitle, obj in scene["actors"]:
        x = positions[key]
        pieces.extend([f'<g class="actor" data-card="object:{obj}" role="button" tabindex="0" aria-label="Explain {html.escape(label, quote=True)}">',
                       f'<rect x="{x-79}" y="16" width="158" height="80" rx="8" class="actor-box"/>',
                       f'<text x="{x}" y="43" text-anchor="middle" class="actor-label">{html.escape(label)}</text>',
                       f'<text x="{x}" y="66" text-anchor="middle" class="actor-subtitle">{html.escape(subtitle)}</text>',
                       f'<text x="{x}" y="84" text-anchor="middle" class="object-hint">inspect object ↗</text>',
                       '</g>', f'<line x1="{x}" x2="{x}" y1="103" y2="284" class="lifeline"/>'])
    for index, step in enumerate(scene["steps"]):
        sender, receiver = positions[step["from"]], positions[step["to"]]
        y = 149 + (index % 3) * 42
        path = (f'M{sender},{y} h54 v26 h-54' if sender == receiver else f'M{sender},{y} H{receiver}')
        active = ' active' if index == 0 else ''
        pieces.append(f'<g class="sequence-event{active}" data-frame="{index}" aria-hidden="{str(index != 0).lower()}">'
                      f'<path d="{path}" data-motion-path class="call-arrow" marker-end="url(#arrow-{scene["id"]})"/>'
                      f'<rect x="{min(sender,receiver)-18}" y="{y-29}" width="36" height="22" rx="5" class="step-number-box"/>'
                      f'<text x="{min(sender,receiver)}" y="{y-13}" text-anchor="middle" class="step-number-text">{index+1}</text></g>')
    pieces.append('<g class="sequence-packet" aria-hidden="true"><circle r="12"/><text text-anchor="middle" y="4">R1</text></g></svg>')
    return "".join(pieces)


def step_body(step):
    fields = html.escape(json.dumps(step["inputs"], indent=2, ensure_ascii=False))
    logic = "".join('<li>' + item + '</li>' for item in step["logic"])
    sources = " · ".join('{{ref:' + key + '}}' for key in step.get("refs", []))
    docs = " · ".join('{{doc:' + key + '}}' for key in step.get("docs", []))
    return (f'<p>{step["summary"]}</p><h4>Inputs and example objects</h4><pre class="object-values">{fields}</pre>'
            f'<h4>What happens</h4><ol>{logic}</ol>'
            f'<p class="result"><strong>Result:</strong> {step["result"]}</p>'
            f'<p class="scale-note"><strong>At scale:</strong> {step["scale"]}</p>'
            f'<p class="sources">{sources}</p><p class="sources">{docs}</p>')


def render_scenes(scenes):
    output = []
    for number, scene in enumerate(scenes, 1):
        steps = []
        library = []
        for index, step in enumerate(scene["steps"]):
            key = scene["id"] + ':' + str(index)
            steps.append(f'<button type="button" class="step-label" data-story-step="{index}" data-card="step:{key}" aria-controls="reading-card" aria-expanded="false" aria-pressed="{str(index == 0).lower()}"><span>{index+1:02d}</span>{step["title"]}</button>')
            library.append(f'<details id="logic-{scene["id"]}-{index}" data-reading="step:{key}"><summary>{index+1} · {step["title"]}</summary><div class="reading-body">{step_body(step)}</div></details>')
        first = scene["steps"][0]
        opened = ' open' if number == 1 else ''
        ascii_diagram = html.escape(scene["ascii"])
        output.append(f'''<details class="scene" id="scene-{scene['id']}" data-scene="{scene['id']}"{opened}>
<summary><span class="section-number">{number:02d}</span><span><strong>{scene['title']}</strong><small>{scene['subtitle']}</small></span><span class="badge">SOURCE + BOUNDARY</span></summary>
<div class="scene-body"><p>{scene['intro']}</p>
<div class="sequence-scroll">{sequence_svg(scene)}</div>
<div class="scene-controls" data-interactive hidden><button type="button" data-action="previous">Back</button><button type="button" data-action="next">Step</button><button type="button" data-action="play" aria-pressed="false">Play</button><button type="button" data-action="replay">Replay</button><label>Speed <select data-speed><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><output data-progress>1 / {len(scene['steps'])}</output></div>
<div class="scene-note"><p class="eyebrow" data-step-evidence>{first['evidence']} · values are EXAMPLE</p><h3 data-step-title>{first['title']}</h3><p data-step-summary>{first['summary']}</p><p class="result" data-step-result>{first['result']}</p></div>
<div class="step-navigation" data-interactive hidden>{''.join(steps)}</div>
<details class="text-fallback"><summary>Portable sequence and ownership</summary><pre>{ascii_diagram}</pre></details>
<div class="reading-library"><details><summary>Read every step explanation</summary>{''.join(library)}</details></div>
</div></details>''')
    return "\n".join(output)


def render_objects(objects):
    output = []
    for key, record in objects.items():
        fields = html.escape(json.dumps(record["fields"], indent=2, ensure_ascii=False))
        body = ('<p>' + record["intro"] + '</p><pre class="object-values">' + fields + '</pre><ol>'
                + ''.join('<li>' + item + '</li>' for item in record["logic"]) + '</ol>'
                + '<p class="sources">' + ' · '.join('{{ref:' + ref + '}}' for ref in record.get("refs", [])) + '</p>'
                + '<p class="sources">' + ' · '.join('{{doc:' + ref + '}}' for ref in record.get("docs", [])) + '</p>')
        output.append(f'<details id="object-{key}" data-reading="object:{key}"><summary>{record["title"]}</summary><div class="reading-body">{body}</div></details>')
    return '<div class="reading-library">' + ''.join(output) + '</div>'


def build_os_systems(data, task, src, span_finder):
    from os_systems_content import DEEP_DIVES, OBJECTS, SCENES, WORKED_EXAMPLES, systems_map

    repo = task.parents[3]
    files = {}
    for path in sorted({FILES[ref["alias"]] for ref in REFS.values()}):
        if path in data["files"]:
            files[path] = data["files"][path]
            continue
        raw = (repo / path).read_bytes()
        pinned = subprocess.check_output(["git", "show", data["revision"] + ':' + path], cwd=repo)
        if raw != pinned:
            raise ValueError("OS source differs from recorded revision: " + path)
        destination = task / "sources" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        files[path] = dict(path=path, snapshot='sources/' + path,
                           sha256=hashlib.sha256(raw).hexdigest(), text=raw.decode())
    refs = {}
    for key, spec in REFS.items():
        path = FILES[spec["alias"]]
        raw = files[path]["text"]
        start, end = span_finder(raw, spec["symbol"])
        code = ''.join(raw.splitlines(keepends=True)[start-1:end])
        line = start
        if needle := spec.get("needle"):
            if code.count(needle) != 1:
                raise ValueError(f"OS source target must match once: {key}: {needle}")
            line += code[:code.index(needle)].count('\n')
        refs[key] = {**spec, "file": path, "start": start, "end": end, "line": line,
                     "code": code, "sha256": hashlib.sha256(code.encode()).hexdigest()}
    research = json.loads((task / "evidence/os_systems_research.json").read_text())
    docs = {record["id"]: record for record in research["records"]}
    for record in docs.values():
        if hashlib.sha256((task / record["snapshot"]).read_bytes()).hexdigest() != record["snapshot_sha256"]:
            raise ValueError("Changed research extract: " + record["id"])
    evidence = json.loads((task / "evidence/llama2_7b_memory_planning_provenance.json").read_text())
    config = json.loads((task / evidence["public_mirror"]["snapshot"]).read_text())
    example = dict(weightBytes=evidence["official"]["total_selected_weight_bytes"],
                   layers=config["num_hidden_layers"], kvHeads=config["num_key_value_heads"],
                   headDim=config["hidden_size"] // config["num_attention_heads"],
                   cacheBytes=2, pageSize=128, maxContext=config["max_position_embeddings"])
    example["tokenBytes"] = 2 * example["layers"] * example["kvHeads"] * example["headDim"] * example["cacheBytes"]
    example["pageBytes"] = example["tokenBytes"] * example["pageSize"]

    template = (src / "os_systems.html").read_text()
    template = template.replace('{{serve-stories}}', render_stories('systems'))
    template = template.replace('{{serve-legend}}', render_legend())
    template = template.replace('{{story-contract}}', story_contract())
    template = template.replace('{{systems-map}}', systems_map())
    template = template.replace('{{scenes}}', render_scenes(SCENES))
    template = template.replace('{{objects}}', render_objects(OBJECTS))
    template = template.replace('{{deep-dives}}', DEEP_DIVES)
    template = template.replace('{{worked-examples}}', WORKED_EXAMPLES)
    used = set()

    def source_link(key, label=None):
        ref = refs[key]
        used.add(key)
        return (f'<a data-ref="{key}" href="../sources/{ref["file"]}#L{ref["line"]}">'
                f'{html.escape(label or ref["label"])} <small>[{ref["alias"]}:L{ref["line"]}]</small></a>')

    directory = ''.join('<li>' + source_link(key) + '</li>' for key in refs)
    template = template.replace('{{source-directory}}', directory)
    cells = []
    for key in CODE_CELLS:
        ref = refs[key]
        rows = '\n'.join(f'<span class="code-line" data-line="{line}"><span class="line-number" aria-hidden="true">{line}</span><span class="line-text">{html.escape(text)}</span></span>'
                         for line, text in enumerate(ref["code"].splitlines(), ref["start"]))
        cells.append(f'<details class="code-cell" id="code-{key}" data-cell="{key}"><summary>Complete source · {html.escape(ref["symbol"])}</summary><div class="code-meta"><code>{ref["file"]}</code> · L{ref["start"]}–L{ref["end"]}</div><div class="code-tools"><button data-copy="{key}" data-interactive hidden>Copy definition</button>{source_link(key, "Source explorer")}<a href="../sources/{ref["file"]}" download>Download full file</a></div><pre class="source-code" tabindex="0"><code>{rows}</code></pre></details>')
    template = template.replace('{{source-cells}}', '\n'.join(cells))
    template = re.sub(r'\{\{ref:([a-z_0-9]+)(?:\|([^}]+))?\}\}', lambda match: source_link(match[1], match[2]), template)

    def doc_link(key, label=None):
        return f'<a href="{html.escape(docs[key]["url"], quote=True)}" target="_blank" rel="noopener">{html.escape(label or docs[key]["title"])}</a>'

    template = re.sub(r'\{\{doc:([a-z_0-9]+)(?:\|([^}]+))?\}\}', lambda match: doc_link(match[1], match[2]), template)
    research_rows = []
    for record in docs.values():
        quotes = ''.join('<blockquote>' + html.escape(paragraph) + '</blockquote>' for paragraph in record["paragraphs"])
        research_rows.append(f'<details id="research-{record["id"]}"><summary>{html.escape(record["title"])}</summary><p>{doc_link(record["id"])} · retrieved {research["retrieved_at"][:10]} · <a href="../{record["snapshot"]}" download>Saved extract</a></p>{quotes}</details>')
    template = template.replace('{{research}}', '\n'.join(research_rows))
    template = template.replace('{{revision}}', data["revision"]).replace('{{revision-short}}', data["revision"][:12])
    template = template.replace('{{weight-gib}}', f'{example["weightBytes"] / 1024**3:.6f}')
    template = template.replace('{{source-count}}', str(len(refs))).replace('{{research-count}}', str(len(docs)))
    if '{{' in template:
        raise ValueError("Unfilled OS systems template marker")
    viewer = dict(revision=data["revision"], files=files, references=refs, example=example,
                  scenes=SCENES, objects=OBJECTS, cells=CODE_CELLS,
                  research=research, scope="Linux + NVIDIA; one independent GPU per lab replica")
    payload = json.dumps(viewer, ensure_ascii=False).replace('<', '\\u003c')
    template = template.replace('<!-- OS_DATA -->', '<script id="os-data" type="application/json">' + payload + '</script>')
    template = template.replace('<!-- OS_CSS -->', '<style>' + (src / "os_systems.css").read_text() + '\n' + (src / "serve_story.css").read_text() + '</style>')
    scripts = ('os_systems_lab.js', 'os_systems.js', 'serve_story_model.js', 'serve_story.js')
    template = template.replace('<!-- OS_JS -->', ''.join('<script>' + (src / name).read_text() + '</script>' for name in scripts))
    (task / "link_htmls" / PAGE_NAME).write_text(template)
    manifest = dict(revision=data["revision"], page='link_htmls/' + PAGE_NAME,
                    references={key: {k: v for k, v in ref.items() if k != "code"} for key, ref in refs.items()},
                    files=[{key: value for key, value in record.items() if key != "text"} for record in files.values()],
                    complete_code_cells=CODE_CELLS, example=example,
                    scenes={scene["id"]: len(scene["steps"]) for scene in SCENES},
                    objects=list(OBJECTS), research_count=len(docs),
                    evidence_files=[dict(path='evidence/os_systems_research.json', sha256=hashlib.sha256((task / 'evidence/os_systems_research.json').read_bytes()).hexdigest()),
                                    dict(path='evidence/llama2_7b_memory_planning_provenance.json', sha256=hashlib.sha256((task / 'evidence/llama2_7b_memory_planning_provenance.json').read_bytes()).hexdigest())])
    (task / "os_systems_manifest.json").write_text(json.dumps(manifest, indent=2) + '\n')
    companion = ["# MAX Serve: Linux, GPU drivers and serving at scale\n",
                 f"[Interactive systems lab](link_htmls/{PAGE_NAME}). SOURCE revision: `{data['revision']}`.\n",
                 "Linux + NVIDIA. Source calls and public-reference mechanisms accompany an explicitly illustrative, deterministic lab.\n",
                 '\n\n'.join(story_markdown('systems'))]
    for scene in SCENES:
        companion += ['\n## ' + scene["title"] + '\n', scene["intro"] + '\n', '```text\n' + scene["ascii"] + '\n```\n']
        for index, step in enumerate(scene["steps"], 1):
            companion += [f'\n### {index}. {step["title"]}\n', step["summary"] + '\n',
                          '```json\n' + json.dumps(step["inputs"], indent=2) + '\n```\n']
            companion += [f'{i}. {re.sub("<[^>]*>", "", item)}\n' for i, item in enumerate(step["logic"], 1)]
            companion += ['\nResult: ' + re.sub('<[^>]*>', '', step["result"]) + '\n',
                          'At scale: ' + re.sub('<[^>]*>', '', step["scale"]) + '\n']
            companion += ['\n' + ' · '.join(f'[{refs[key]["label"]}](sources/{refs[key]["file"]}#L{refs[key]["line"]})' for key in step.get("refs", [])) + '\n']
    (task / "OS_SYSTEMS_GUIDE.md").write_text('\n'.join(companion).rstrip() + '\n')
    return len(refs), len(CODE_CELLS)
