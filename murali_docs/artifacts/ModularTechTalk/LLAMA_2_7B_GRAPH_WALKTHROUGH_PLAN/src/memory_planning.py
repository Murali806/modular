"""Build the offline memory-planning QnA with complete, pinned source."""

from __future__ import annotations

import hashlib
import html
import json
import re

from call_stack_refs import FILES as STACK_FILES, REFS as STACK_REFS

PAGE_NAME = "llama2_7b_memory_planning_and_memory_plan.html"
BASE = "max/python/max/"
FILES = {**STACK_FILES, **{key: BASE + value for key, value in {
    "DEVICES": "pipelines/lib/device_specs.py",
    "DRIVER": "driver/driver.py",
    "RUNTIME": "pipelines/lib/pipeline_runtime_config.py",
    "ARCH_CONFIG": "pipelines/lib/interfaces/arch_config.py",
    "KV_CONFIG": "pipelines/kv_cache/config.py",
    "PLANNER": "pipelines/kv_cache/memory_planner.py",
    "KV_REGISTRY": "pipelines/kv_cache/registry.py",
    "KV_PARAMS": "nn/kv_cache/cache_params.py",
    "JENGA": "pipelines/kv_cache/paged_kv_cache/jenga_cache_manager.py",
    "GEOMETRY": "pipelines/kv_cache/paged_kv_cache/jenga_block_pool.py",
}.items()}}


def definition(alias, symbol, label=None, needle=None):
    result = dict(alias=alias, symbol=symbol, label=label or symbol + "()", kind="definition")
    if needle:
        result["needle"] = needle
    return result


REFS = {key: STACK_REFS[key] for key in (
    "memory", "memory_call", "factory", "factory_call", "config", "config_call",
    "from_args", "hf_load", "autoconfig", "llama_arch", "initialize", "memory_init_call",
    "partial", "serve", "lifespan", "start_worker", "spawn", "ready", "connected",
    "worker_run", "invoke", "pipeline_init", "weight_paths", "load_weights",
    "llama_init", "finalized", "finalize", "compile_init", "alive", "http",
)}
REFS.update({
    "plan_type": definition("MEMORY", "MemoryPlan", "MemoryPlan (frozen dataclass)"),
    "plan_sizes": definition("MEMORY", "MemoryEstimator.plan_from_sizes"),
    "free_memory": definition("MEMORY", "MemoryEstimator._free_memory"),
    "batch_infer": definition("MEMORY", "MemoryEstimator._infer_optimal_batch_size"),
    "length_fit": definition("MEMORY", "MemoryEstimator._find_valid_max_length"),
    "capacity": definition("MEMORY", "MemoryEstimator._max_supported_sequence_length"),
    "cache_size": definition("MEMORY", "MemoryEstimator._calculate_kv_cache_size"),
    "context_cap": definition("MEMORY", "MemoryEstimator._resolve_max_batch_total_tokens"),
    "context_cap_default": definition("LOOKUP", "SupportedArchitecture", "SupportedArchitecture.requires_max_batch_context_length = False", "    requires_max_batch_context_length: bool = False"),
    "vision": definition("MEMORY", "MemoryEstimator._reserve_vision_cache_memory"),
    "default_batch": definition("MEMORY", "_DEFAULT_BATCH_SIZE", "_DEFAULT_BATCH_SIZE = 512"),
    "device_defaults": definition("DEVICES", "_default_device_specs"),
    "normalize_devices": definition("DEVICES", "normalize_device_specs_input"),
    "device_specs": definition("DEVICES", "device_specs_from_normalized_device_handle"),
    "devices_flag": definition("FLAGS", "pipeline_config_options", "CLI device_specs mapping", "kwargs[\"device_specs\"] = DevicesOptionType.device_specs(devices)"),
    "load_devices": definition("DRIVER", "load_devices"),
    "load_device": definition("DRIVER", "load_device"),
    "scan_devices": definition("DRIVER", "scan_available_devices"),
    "runtime_defaults": definition("RUNTIME", "PipelineRuntimeConfig", "PipelineRuntimeConfig (batch defaults)", "    max_batch_size: int | None = Field("),
    "input_token_default": definition("RUNTIME", "DEFAULT_MAX_BATCH_INPUT_TOKENS", "DEFAULT_MAX_BATCH_INPUT_TOKENS = 8192"),
    "length_policy": definition("ARCH_CONFIG", "ArchConfigWithBoundedMaxSeqLen.calculate_max_seq_len"),
    "resolve_length": definition("CONFIG", "_resolve_models_max_length"),
    "initialize_config": definition("LLAMA_CFG", "Llama3Config.initialize_from_config"),
    "llama_config_type": definition("LLAMA_CFG", "Llama3Config", "Llama3Config.SUPPORTED_ENCODINGS", "    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"),
    "kv_construct": definition("LLAMA_CFG", "Llama3Config.construct_kv_params"),
    "head_dim": definition("ARCH_CONFIG", "ArchConfigWithStoredKVParams.get_head_dim"),
    "kv_defaults": definition("KV_CONFIG", "KVCacheConfig", "KVCacheConfig (page size and utilization)", "    kv_cache_page_size: int = Field("),
    "kv_dtype": definition("KV_CONFIG", "cache_dtype_for_encoding"),
    "kv_params": definition("KV_CONFIG", "KVCacheConfig.to_params"),
    "weights_size": definition("MODEL_CFG", "MAXModelConfig.weights_size"),
    "repo_size": definition("WEIGHTS", "HuggingFaceRepo.size_of"),
    "planner": definition("PLANNER", "PagedMemoryPlanner", "PagedMemoryPlanner (Llama declaration)"),
    "weight_estimate": definition("PLANNER", "MemoryPlanner.estimate_weights_size"),
    "activation_estimate": definition("PLANNER", "PagedMemoryPlanner.estimate_activation_memory"),
    "planner_batch": definition("PLANNER", "MemoryPlanner.infer_max_batch_size"),
    "signal_estimate": definition("PLANNER", "MemoryPlanner.estimate_signal_buffer_memory"),
    "signal_config": definition("CONFIG", "PipelineConfig.estimate_signal_buffer_memory"),
    "manager_select": definition("KV_REGISTRY", "_use_jenga_kv_cache"),
    "manager_size": definition("KV_REGISTRY", "kv_cache_memory_size"),
    "manager_capacity": definition("KV_REGISTRY", "max_seq_len_fitting_in_cache"),
    "manager_load": definition("KV_REGISTRY", "load_kv_manager"),
    "generic_blocks": definition("KV_PARAMS", "compute_num_device_blocks"),
    "generic_size": definition("KV_PARAMS", "estimated_memory_size"),
    "generic_capacity": definition("KV_PARAMS", "compute_max_seq_len_fitting_in_cache"),
    "bytes_per_block": definition("KV_PARAMS", "KVCacheParams.bytes_per_block", "KVCacheParams.bytes_per_block (property)"),
    "value_block": definition("KV_PARAMS", "KVCacheParams.bytes_per_value_block", "KVCacheParams.bytes_per_value_block (property)"),
    "leaves": definition("KV_PARAMS", "KVCacheParams.leaves"),
    "jenga_geometry": definition("JENGA", "JengaKVCacheManager._plan_geometry"),
    "jenga_size": definition("JENGA", "JengaKVCacheManager.memory_size"),
    "jenga_capacity": definition("JENGA", "JengaKVCacheManager.max_seq_len_fitting_in_cache"),
    "geometry": definition("GEOMETRY", "plan_jenga_geometry"),
    "encoding": definition("MODEL_CFG", "_resolve_weights_and_encoding"),
    "encoding_infer": definition("MODEL_CFG", "_infer_quantization_encoding"),
    "encoding_select": definition("MODEL_CFG", "_select_encoding_and_dtype_cast"),
    "repo_encodings": definition("WEIGHTS", "HuggingFaceRepo.supported_encodings", "HuggingFaceRepo.supported_encodings (property)"),
})


def build_memory_planning(data, task, src, span_finder):
    refs = {}
    for key, spec in REFS.items():
        file = FILES[spec["alias"]]
        text = data["files"][file]["text"]
        start, end = span_finder(text, spec["symbol"])
        code = "".join(text.splitlines(keepends=True)[start - 1:end])
        line = start
        if needle := spec.get("needle"):
            if code.count(needle) != 1:
                raise ValueError(f"Memory source target must match once: {key}: {needle}")
            line += code[:code.index(needle)].count("\n")
        refs[key] = {**spec, "file": file, "start": start, "end": end, "line": line,
                     "code": code, "sha256": hashlib.sha256(code.encode()).hexdigest()}

    used_refs, used_cells = set(), set()
    template = (src / "memory_planning.html").read_text()

    def render_source_link(key, label=None):
        ref = refs[key]
        used_refs.add(key)
        label = label or ref["label"]
        return (f'<a class="source-link" data-ref="{key}" '
                f'href="../sources/{ref["file"]}#L{ref["line"]}" '
                f'title="{html.escape(ref["file"])} · L{ref["line"]} · complete {html.escape(ref["symbol"])}">'
                f'{html.escape(label)} <span class="source-location">[{ref["alias"]}:L{ref["line"]}]</span></a>')

    template = re.sub(r"\{\{ref:([a-z_0-9]+)(?:\|([^}]+))?\}\}",
                      lambda match: render_source_link(match[1], match[2]), template)

    def code_cell(match):
        key = match[1]
        ref = refs[key]
        used_refs.add(key)
        used_cells.add(key)
        rows = "\n".join(
            f'<span class="code-line" data-line="{number}"><span class="line-number" aria-hidden="true">{number}</span>'
            f'<span class="line-text">{html.escape(value)}</span></span>'
            for number, value in enumerate(ref["code"].splitlines(), ref["start"]))
        return f'''<details class="code-cell" id="code-{key}" data-cell="{key}">
<summary><span class="summary-main"><strong>Complete source · {html.escape(ref['symbol'])}</strong><span>{ref['end'] - ref['start'] + 1} original lines · SOURCE</span></span></summary>
<div class="code-meta"><code>{html.escape(ref['file'])}</code><span>L{ref['start']}–L{ref['end']} · {data['revision'][:12]}</span></div>
<div class="code-tools"><button type="button" data-copy="{key}" data-interactive hidden>Copy complete definition</button>
<a href="../sources/{ref['file']}#L{ref['start']}" data-ref="{key}">Source explorer</a>
<a href="../sources/{ref['file']}" download>Download full file</a></div>
<pre class="source-code" tabindex="0" aria-label="Complete original {html.escape(ref['symbol'])}"><code>{rows}</code></pre></details>'''

    template = re.sub(r"\{\{code:([a-z_0-9]+)\}\}", code_cell, template)
    evidence = json.loads((task / "evidence/llama2_7b_memory_planning_provenance.json").read_text())
    config = json.loads((task / evidence["public_mirror"]["snapshot"]).read_text())
    weight_bytes = evidence["official"]["total_selected_weight_bytes"]
    assert weight_bytes == sum(item["size"] for item in evidence["official"]["selected_weight_files"])
    assert config["hidden_size"] % config["num_attention_heads"] == 0
    example = {
        "weightBytes": weight_bytes,
        "layers": config["num_hidden_layers"],
        "kvHeads": config["num_key_value_heads"],
        "queryHeads": config["num_attention_heads"],
        "headDim": config["hidden_size"] // config["num_attention_heads"],
        "maxPosition": config["max_position_embeddings"],
        "cacheBytes": 2, "pageSize": 128, "batch": 4, "length": 4096,
        "freeGiB": 24, "utilization": 0.9,
    }
    # Every registered definition must be reachable through a source link or cell.
    # The collapsed directory also makes the less common helpers discoverable.
    directory = []
    for key, ref in refs.items():
        directory.append('<li>' + render_source_link(key) + '</li>')
    template = template.replace("{{source-directory}}", "\n".join(directory))
    replacements = {
        "{{revision}}": data["revision"],
        "{{revision-short}}": data["revision"][:12],
        "{{mirror-config}}": html.escape(json.dumps(config, indent=2)),
        "{{evidence-date}}": html.escape(evidence["retrieved_at"][:10]),
        "{{official-api}}": html.escape(evidence["official"]["api_url"], quote=True),
        "{{mirror-url}}": html.escape(evidence["public_mirror"]["url"], quote=True),
        "{{weight-gib}}": f"{weight_bytes / 1024**3:.6f}",
        "{{source-count}}": str(len(refs)),
        "{{baseline-plan}}": html.escape(json.dumps({
            "planned_max_batch_size": 4,
            "footprint": weight_bytes + 129 * 64 * 1024**2,
            "planned_max_length": 4096,
            "available_cache_memory": 129 * 64 * 1024**2,
            "device_specs": [{"device_type": "gpu", "id": 0}],
            "planned_max_batch_total_tokens": None,
            "vision_cache_plan": None,
        }, indent=2)),
        "{{page-blocks}}": "".join(f'<span class="page-block" title="Page {i}: 128 tokens">{i:02d}</span>' for i in range(32)),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    if "{{" in template:
        raise ValueError("Unfilled memory-planning template marker")
    files = {ref["file"]: data["files"][ref["file"]] for ref in refs.values()}
    viewer = dict(revision=data["revision"], references=refs, files=files, example=example,
                  evidence=evidence, cells=sorted(used_cells))
    payload = json.dumps(viewer, ensure_ascii=False).replace("<", "\\u003c")
    template = template.replace("<!-- MEMORY_DATA -->", '<script id="memory-data" type="application/json">' + payload + '</script>')
    template = template.replace("<!-- MEMORY_CSS -->", '<style>' + (src / "memory_planning.css").read_text() + '</style>')
    template = template.replace("<!-- MEMORY_JS -->", '<script>' + (src / "memory_planning.js").read_text() + '</script>')
    destination = task / "link_htmls" / PAGE_NAME
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(template)
    manifest = dict(revision=data["revision"], page=f"link_htmls/{PAGE_NAME}",
                    references={key: {k: v for k, v in ref.items() if k != "code"} for key, ref in refs.items()},
                    complete_code_cells=sorted(used_cells), example=example,
                    evidence_files=[dict(path=str(path.relative_to(task)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                    for path in (task / "evidence/llama2_7b_memory_planning_provenance.json", task / evidence["public_mirror"]["snapshot"])])
    (task / "memory_planning_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return len(refs), len(used_cells)
