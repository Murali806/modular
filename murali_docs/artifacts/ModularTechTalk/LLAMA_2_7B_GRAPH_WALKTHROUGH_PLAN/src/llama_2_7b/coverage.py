"""Every Example 04 companion has an explicit Llama adaptation."""
import html
import json
from catalog import OUT, TASK

MAPPING = {
 'fused_attention_online_softmax': ('index attention_rope', 'Whole-model lifecycle; production paged attention and numerical softmax recurrence.'),
 'gpu_artifacts_abi_and_driver': ('gpu_artifacts driver', 'Host/device artifacts, two argument packs, retained function handles and buffer ownership.'),
 'virtual_device_count_and_slots': ('session driver', 'Symbolic/virtual device identity versus physical device context and execution.'),
 'nvml_cuda_mig_and_fork': ('session driver', 'Discovery versus CUDA execution; process and device-scope boundaries.'),
 'asyncrt_runtime_deep_dive': ('asyncrt_values asyncrt_workers asyncrt_execution', 'AsyncValue state machine, queues, await donation, wakeup race and model readiness.'),
 'mlir_context_and_worker_threads': ('session graph_scaffold', 'Compiler pool versus AsyncRT workers, thread-local compiler scope.'),
 'mlir_context_constructor_internals': ('graph_scaffold', 'Context ownership, type/attribute/affine uniquers and thread pool setup.'),
 'tensor_type_to_mlir': ('graph_scaffold inputs_shapes', 'Typed model/KV inputs and dtype/shape/device conversion.'),
 'graph_regions_blocks_and_arguments_qna': ('graph_scaffold decoder_trace', 'Module → GraphOp → region → block → arguments → operations.'),
 'graph_init_context_decorator': ('graph_scaffold', 'Context decorator scope and Graph construction order.'),
 'moduleop_create_deep_dive': ('graph_scaffold', 'Module ownership, GraphOp insertion and CAPI view over the same operation.'),
 'shape_ape_type_explanation': ('graph_scaffold inputs_shapes mo_contracts', 'Symbolic dimension expressions, static values and runtime shape materialization.'),
 'float32_dtype_vs_i64_shape_dimensions': ('graph_scaffold inputs_shapes', 'Element dtype and signed-64-bit shape-parameter representation.'),
 'shape_dimension_simd_and_constant_folding': ('inputs_shapes mo_contracts', 'One-lane shape integer types and conditional folding without tensor-value inference.'),
 'device_ref_attribute_placement_qna': ('graph_scaffold session', 'Placement metadata versus live driver resources.'),
 'graph_weights_and_ordering_chains_qna': ('decoder_trace graph_scaffold mgp_plan', 'Weight registries, mutable KV ordering and execution lifetime.'),
 'graph_enter_context_manager': ('graph_scaffold', 'Graph enter/exit and forward closure tracing for the model path.'),
 'custom_op_g1_b0_state_changes': ('graph_scaffold attention_rope mo_to_map', 'Record production rope/store and paged MHA operations with concrete input contracts.'),
 'session_constructor_and_runtime_state': ('session', 'Device normalization, CPU addition, native config and shared runtime ownership.'),
 'session_num_threads_qna': ('session asyncrt_workers', 'Requested thread hint versus existing CPUDevice; worker/foreign await branches.'),
 'session_custom_extensions_qna': ('build_startup session', 'Built-in P1 packages, empty explicit GraphSpec extensions and possible overlays.'),
 'runtime_config_sources_and_llama_qna': ('build_startup session', 'Actual serving caller, resolved policies and encoding acceptance gate.'),
 'session_device_spec_and_context_qna': ('session driver', 'Target description, runtime context, streams, device handles and retention.'),
 'max_context_creation_and_mlir_qna': ('session graph_scaffold', 'MAX engine context and MLIR compilation context are separate objects.'),
 'runtime_ready_chain_vs_graph_chain_qna': ('graph_scaffold asyncrt_values asyncrt_execution', 'Symbolic ordering, runtime AsyncValue readiness and GPU completion.'),
 'c1_mo_contracts_and_canonicalization': ('mo_contracts', 'Cloned IR1 contracts, pass predicates, shape/layout inference and cleanup.'),
 'c1_mo_to_map_access_and_compute': ('mo_to_map', 'Model tensor operations split into access/view and computation representations.'),
 'c1_map_fusion_and_access_optimization': ('map_optimization', 'Fusion legality around residuals, mutable KV and opaque production kernels.'),
 'c1_map_to_mogg_implementation_calls': ('map_to_mogg', 'Kernel declarations, typed implementation calls and captured accesses.'),
 'c1_mogg_call_conformance_and_outlining': ('mogg_conformance', 'Operand adaptation, parameter bindings and outlined routines.'),
 'c1_mogg_to_mgp_models_buffers_and_chains': ('mgp_plan', 'Executable model, typed buffer slots, dependencies and memory lifetimes.'),
 'c1_mgp_optimization_and_mojo_emission': ('mgp_plan mojo_codegen', 'Ordered memory/code passes, async outlining and host stub generation.'),
 'c1_jit_compilation_and_gpu_toolchain_qna': ('mojo_codegen gpu_artifacts', 'Kernel specialization, cache keys, LLVM, host object and GPU image production.'),
 'post_c1_d1_gpu_code_loading_qna': ('gpu_artifacts initialization driver', 'Compiled bytes versus runtime CUDA module/function handles; lazy first use.'),
 'm1_host_code_loading_memory_and_execution': ('initialization asyncrt_execution lifetime', 'ORC/JITLink, bound resources, initialized model reuse and safe release.'),
}


def build_coverage(chapters, header, footer):
    existing = {c['slug']:c for c in chapters}
    files = sorted((TASK/'link_htmls').glob('max_example_04*.html'))
    observed={p.stem.removeprefix('max_example_04_') for p in files}
    assert observed == set(MAPPING), (observed-set(MAPPING),set(MAPPING)-observed)
    text=header('Coverage and evidence audit','Compare every Example 04 companion with the model-specific explanation.','',chapters)
    text+='<p>The original companions remain available for side-by-side review. The Llama pages adapt the mechanisms to serving order, production attention, persistent KV and separate sampler graphs.</p><table class="coverage-list"><thead><tr><th>Example 04 companion</th><th>Llama lessons</th><th>What is carried forward or changed</th></tr></thead><tbody>'
    rows=[]
    for p in files:
        key=p.stem.removeprefix('max_example_04_'); targets,note=MAPPING[key]
        assert set(targets.split()) <= set(existing), (key, targets)
        links=' · '.join(f'<a href="{t}.html">{html.escape(existing[t]["title"])}</a>' if t in existing else html.escape(t) for t in targets.split())
        text+=f'<tr><td><a href="../{p.name}">{html.escape(key.replace("_"," "))}</a></td><td>{links}</td><td>{html.escape(note)}</td></tr>'
        rows.append(dict(example=p.name,lessons=targets.split(),adaptation=note))
    text+='</tbody></table><h2>Additional model mechanisms</h2><ul><li>Exact dimensions and parameter accounting; FP16 acceptance gate.</li><li>32-layer tracing and named weight/schema binding.</li><li>Rotary positions, rank-6 KV pages, physical stride and page ownership.</li><li>Ragged prefill/decode, logits selection, separate sampler and tracked token copy.</li><li>Persistent versus execution/request lifetime and compiler cache policies.</li></ul><h2>Evidence limits</h2><p>SOURCE proves source contracts and control flow. UI checks and JavaScript calculations prove reader behavior only. No model compilation, GPU inference, exact emitted IR, selected target kernel or timing measurement is claimed.</p>'
    (OUT/'coverage.html').write_text(text+footer())
    (OUT/'coverage.json').write_text(json.dumps(rows,indent=2)+'\n')
