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
    "input_token_setting": definition("RUNTIME", "PipelineRuntimeConfig", "runtime.max_batch_input_tokens (prefill target)", "    max_batch_input_tokens: int = Field("),
    "resident_token_setting": definition("RUNTIME", "PipelineRuntimeConfig", "runtime.max_batch_total_tokens (page-aligned contexts)", "    max_batch_total_tokens: int | None = Field("),
    "batch_input_clamp": definition("MEMORY", "MemoryEstimator.plan_from_sizes", "Cap batch size by max_batch_input_tokens", "        if max_batch_size > pipeline_config.runtime.max_batch_input_tokens:"),
    "oom_search_gate": definition("MEMORY", "MemoryEstimator.plan_from_sizes", "Guard the default-length OOM search", "        if int(total_size) > free_memory and not user_provided_max_length:"),
    "capacity_clamp": definition("MEMORY", "MemoryEstimator.plan_from_sizes", "Apply the committed-cache capacity bound", "        if kv_capacity := cls._max_supported_sequence_length("),
    "length_policy": definition("ARCH_CONFIG", "ArchConfigWithBoundedMaxSeqLen.calculate_max_seq_len"),
    "resolve_length": definition("CONFIG", "_resolve_models_max_length"),
    "initialize_config": definition("LLAMA_CFG", "Llama3Config.initialize_from_config"),
    "planner_model_access": definition("MEMORY", "MemoryEstimator.plan", "Read pipeline_config.model", "        model_config = pipeline_config.model"),
    "pipeline_main_model": definition("CONFIG", "PipelineConfig.model", "PipelineConfig.model (main-model property)"),
    "initialize_hf_access": definition("LLAMA_CFG", "Llama3Config.initialize", "Read model_config.huggingface_config", "        huggingface_config = model_config.huggingface_config"),
    "initialize_delegate": definition("LLAMA_CFG", "Llama3Config.initialize", "Delegate to initialize_from_config", "        return cls.initialize_from_config("),
    "initialize_settings": definition("LLAMA_CFG", "Llama3Config.initialize_from_config", "Read model_config.kv_cache and encoding", "        kv_cache_config = model_config.kv_cache"),
    "initialize_result": definition("LLAMA_CFG", "Llama3Config.initialize_from_config", "Construct the returned Llama3Config", "        return cls("),
    "attention_multiplier": definition("LLAMA_CFG", "Llama3Config.calculate_attention_multiplier"),
    "kv_to_params_call": definition("LLAMA_CFG", "Llama3Config.construct_kv_params", "Call kv_cache_config.to_params", "        return kv_cache_config.to_params("),
    "mha_params_type": definition("KV_PARAMS", "MHAKVCacheParams", "MHAKVCacheParams (KVCacheParams subclass)"),
    "planner_config_call": definition("MEMORY", "MemoryEstimator.plan", "Pass arch_config to the memory planner", "            planner = arch.memory_planner(arch_config)"),
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


def initialization_sequence(refs, example):
    """Render the initialization calls with local source and object links."""
    def source_link(key, content, step=None):
        ref = refs[key]
        trigger = (f'data-init-step="{step}" aria-controls="init-step-card" '
                   'aria-expanded="false" aria-describedby="init-step-hint" ' if step else '')
        return (f'<a class="init-source" data-ref="{key}" '
                + trigger +
                f'href="../sources/{ref["file"]}#L{ref["line"]}" '
                f'aria-label="{html.escape(ref["label"], quote=True)}; source line {ref["line"]}">'
                + content + '</a>')

    def object_link(key, content):
        return f'<a class="init-object" href="#init-object-{key}">{content}</a>'

    actors = [
        ("planner", "MemoryEstimator.plan()", "API parent / caller", "memory"),
        ("initializer", "Llama3Config", "class methods", "initialize"),
        ("pipeline", "P1 · pipeline_config", "PipelineConfig", None),
        ("model", "M1 · model_config", "MAXModelConfig", None),
        ("hf", "H1 · huggingface_config", "AutoConfig / metadata", None),
        ("kv-config", "K1 · kv_cache_config", "KVCacheConfig", None),
        ("arch", "A1 · arch_config", "Llama3Config · created at 11", None),
    ]
    positions = {actor[0]: 115 + index * 170 for index, actor in enumerate(actors)}
    layers, heads, width = example["layers"], example["kvHeads"], example["headDim"]
    hidden = example["queryHeads"] * width
    length, page = example["length"], example["pageSize"]
    events = [
        ("planner", "pipeline", "Read the main model from PipelineConfig", f"P1.model = P1.models['main'] = M1; M1.max_length = {length}", "planner_model_access", "pipeline", False),
        ("planner", "initializer", "Call the selected architecture's initializer", f"arch.config = Llama3Config; initialize(P1, model_config=M1, max_seq_len={length})", "memory_init_call", "model", False),
        ("initializer", "model", "Select MAXModelConfig and read its Hugging Face metadata", "M1.huggingface_config = H1; omitted model_config uses P1.model; missing H1 raises ValueError", "initialize_hf_access", "hf", False),
        ("initializer", "initializer", "Delegate to initialize_from_config()", f"initialize_from_config(P1, H1, M1, max_seq_len={length})", "initialize_delegate", "model", False),
        ("initializer", "model", "Read the model's encoding, KV settings and devices", "M1: quantization_encoding='bfloat16'; kv_cache=K1; device_specs=[gpu:0]; data_parallel_degree=1", "initialize_settings", "model", False),
        ("initializer", "initializer", "Resolve model dtype, cache dtype and DeviceRefs", "dtype=BF16; cache_dtype=BF16 (2 bytes); devices=[DeviceRef('gpu', 0)]", "kv_dtype", "kv-config", False),
        ("initializer", "hf", "Read model dimensions and derive the head width", f"H1: layers={layers}; hidden={hidden}; query_heads={example['queryHeads']}; KV heads={heads}; head_dim={hidden}/{example['queryHeads']}={width}", "head_dim", "hf", False),
        ("initializer", "initializer", "Construct the KV-cache parameters", "construct_kv_params(H1, P1, [gpu:0], K1, BF16); DP=1; speculative=None -> num_draft_tokens=0", "kv_construct", "kv-params", False),
        ("initializer", "kv-config", "Call KVCacheConfig.to_params()", f"dtype=BF16; n_kv_heads={heads}; head_dim={width}; num_layers={layers}; allow_kv_head_replication=False", "kv_to_params_call", "kv-config", False),
        ("kv-config", "initializer", "Return V1: MHAKVCacheParams, a KVCacheParams subclass", f"page_size={page}; slots_per_page={page}; KV heads/device={heads}; devices=[gpu:0]; DP=1", "kv_params", "kv-params", True),
        ("initializer", "arch", "Create A1: Llama3Config", f"hidden={hidden}; layers={layers}; max_seq_len={length}; dtype=BF16; kv_params=V1", "initialize_result", "arch", False),
        ("initializer", "planner", "Return the architecture config to MemoryEstimator.plan()", "arch_config=A1; arch_config.kv_params is V1; the planner receives the new architecture object", "memory_init_call", "arch", True),
    ]
    height = 124 + len(events) * 86
    pieces = [f'<svg viewBox="0 0 1280 {height}" role="group" aria-labelledby="init-sequence-title init-sequence-desc">',
              '<title id="init-sequence-title">Llama3Config initialization: calls, source and example objects</title>',
              '<desc id="init-sequence-desc">Twelve ordered steps in the API parent. The planner supplies a PipelineConfig and MAXModelConfig; Llama3Config reads Hugging Face metadata and cache settings, constructs MHAKVCacheParams, and returns the architecture config. Step links open exact source; object links reveal example fields.</desc>',
              '<defs><marker id="init-sequence-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" class="init-arrowhead"/></marker></defs>']
    for key, title, kind, ref_key in actors:
        x = positions[key]
        lifeline_start = 120 + 10 * 86 + 62 if key == "arch" else 104
        pieces.append(f'<line x1="{x}" x2="{x}" y1="{lifeline_start}" y2="{height - 12}" class="init-lifeline"/>')
        header = (f'<rect x="{x - 80}" y="14" width="160" height="80" rx="7" class="init-actor-box"/>'
                  f'<text x="{x}" y="44" text-anchor="middle" class="init-actor-label">{html.escape(title)}</text>'
                  f'<text x="{x}" y="68" text-anchor="middle" class="init-actor-type">{html.escape(kind)}</text>')
        pieces.append(source_link(ref_key, header) if ref_key else object_link(key, header))
    for index, (sender, receiver, title, value, ref_key, obj, reply) in enumerate(events):
        y = 120 + index * 86
        left, right = positions[sender], positions[receiver]
        path = (f'M{left},{y + 62} h60 v14 h-60' if left == right
                else f'M{left},{y + 62} H{right}')
        ref = refs[ref_key]
        source_content = (f'<text x="59" y="{y + 17}" class="init-step-label">{html.escape(title)}</text>'
                          f'<text x="1244" y="{y + 17}" text-anchor="end" class="init-source-location">[{ref["alias"]}:L{ref["line"]}]</text>'
                          f'<path d="{path}" class="init-call{" init-return" if reply else ""}" marker-end="url(#init-sequence-arrow)"/>')
        pieces.extend([f'<g data-step="{index + 1}">',
                       f'<rect x="26" y="{y - 4}" width="1228" height="49" rx="5" class="init-message-box"/>',
                       f'<text x="39" y="{y + 17}" text-anchor="middle" class="init-message-number">{index + 1}</text>',
                       source_link(ref_key, source_content, step=index + 1),
                       object_link(obj, f'<text x="59" y="{y + 37}" class="init-value">{html.escape(value)}</text>'),
                       '</g>'])
    return "".join(pieces) + '</svg>'


def initialization_objects(config, example, source_link):
    """Show selected input and output fields for the diagram's BF16 example."""
    devices = ['DeviceRef("gpu", 0)']
    records = [
        ("pipeline", "P1 · pipeline_config: PipelineConfig", "config", {
            "models": {"main": "M1"}, "model (property)": "M1",
            "runtime.max_batch_size": example["batch"], "speculative": None,
        }),
        ("model", "M1 · model_config: MAXModelConfig", "from_args", {
            "model_path": "meta-llama/Llama-2-7b-hf", "huggingface_config": "H1",
            "max_length": example["length"], "quantization_encoding": "bfloat16",
            "device_specs": [{"device_type": "gpu", "id": 0}],
            "kv_cache": "K1", "data_parallel_degree": 1, "use_subgraphs": False,
        }),
        ("hf", "H1 · huggingface_config: AutoConfig", "hf_load", {
            key: config[key] for key in ("architectures", "hidden_size", "num_hidden_layers",
                                        "num_attention_heads", "num_key_value_heads",
                                        "intermediate_size", "vocab_size", "max_position_embeddings",
                                        "rope_scaling", "torch_dtype")
        }),
        ("kv-config", "K1 · kv_cache_config: KVCacheConfig", "kv_defaults", {
            "kv_cache_page_size": example["pageSize"], "kv_cache_format": None,
            "enable_prefix_caching": True, "device_memory_utilization": example["utilization"],
        }),
        ("kv-params", "V1 · kv_params: MHAKVCacheParams", "mha_params_type", {
            "dtype": "DType.bfloat16", "n_kv_heads": example["kvHeads"],
            "head_dim": example["headDim"], "num_layers": example["layers"],
            "page_size": example["pageSize"], "slots_per_page": example["pageSize"],
            "devices": devices, "data_parallel_degree": 1, "allow_kv_head_replication": False,
            "enable_prefix_caching": True, "speculative_method": None, "num_draft_tokens": 0,
            "n_kv_heads_per_device (property)": example["kvHeads"],
        }),
        ("arch", "A1 · arch_config: Llama3Config", "initialize_result", {
            "hidden_size": config["hidden_size"], "num_hidden_layers": example["layers"],
            "num_attention_heads": example["queryHeads"], "num_key_value_heads": example["kvHeads"],
            "max_seq_len": example["length"], "intermediate_size": config["intermediate_size"],
            "vocab_size": config["vocab_size"], "dtype": "DType.bfloat16",
            "quantization_encoding": "bfloat16", "devices": devices, "kv_params": "V1",
            "rope_scaling_params": None,
            "longrope_scaling_params": None, "attention_multiplier": (1 / example["headDim"]) ** 0.5,
            "data_parallel_degree": 1,
        }),
    ]
    return "\n".join(
        f'<details class="explanation init-object-record" id="init-object-{key}">'
        f'<summary><strong>{html.escape(title)}</strong></summary><div class="detail-body">'
        f'<p class="sources">{source_link(ref_key)}</p>'
        f'<pre tabindex="0" aria-label="Example fields for {html.escape(title, quote=True)}">'
        f'{html.escape(json.dumps(values, indent=2))}</pre></div></details>'
        for key, title, ref_key, values in records)


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
    template = template.replace("{{initialize-sequence}}", initialization_sequence(refs, example))
    template = template.replace("{{initialize-objects}}", initialization_objects(config, example, render_source_link))
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
