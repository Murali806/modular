# Llama 7B Causal LM Call-Flow File Tree

Evidence: `SOURCE`

Scope: built-in `LlamaForCausalLM`, single-device core path, with optional serving, tensor-parallel, data-parallel, overlap, and speculative-decoding branches identified in place.

```text
modular/
+-- max/python/max/
    +-- _entrypoints/cli/
    |   +-- generate.py
    |   |   +-- Local loop: prompt -> context -> execute -> decode.
    |   |   +-- Functions: stream_context(), stream_text_to_console(), generate_text_for_pipeline(), _with_debug_defaults().
    |   +-- serve/
    |       +-- serve_api_and_model_worker.py
    |           +-- Starts max serve and resolves LlamaForCausalLM.
    |           +-- Functions: serve_api_server_and_model_worker(), _exit_on_signal(), serve().
    +-- serve/
    |   +-- api_server.py
    |   |   +-- Builds FastAPI and connects it to the worker.
    |   |   +-- Functions: fastapi_app(), fastapi_config(), validate_port_is_free(), version().
    |   +-- router/
    |   |   +-- openai_routes.py
    |   |       +-- Converts OpenAI JSON to internal requests.
    |   |       +-- Functions: openai_create_chat_completion(), openai_create_completion(), openai_parse_chat_completion_request(), get_pipeline().
    |   +-- pipelines/
    |   |   +-- llm.py
    |   |   |   +-- Tokenizes requests and streams generated token chunks.
    |   |   |   +-- Functions: TokenGeneratorPipeline.next_token_chunk(), all_tokens(), score(), encode(), _apply_stop_truncation().
    |   |   +-- model_worker.py
    |   |       +-- Builds the compiled pipeline once and runs it.
    |   |       +-- Functions: ModelWorker.run(), ModelWorker.__call__(), get_pipeline_model(), _prime_pinned_memory_cache().
    |   +-- scheduler/
    |       +-- text_generation_scheduler.py
    |       |   +-- Schedules prefill and decode batches.
    |       |   +-- Functions: TokenGenerationScheduler.run_iteration(), _schedule(), _retrieve_pending_requests(), load_text_generation_scheduler().
    |       +-- batch_constructor/
    |           +-- text_batch_constructor.py
    |               +-- Packs requests under token and KV-cache limits.
    |               +-- Functions: enqueue_new_request(), construct_batch(), advance_requests(), release_request(), _add_ce_requests(), _add_tg_requests().
    +-- pipelines/
    |   +-- architectures/
    |   |   +-- __init__.py
    |   |   |   +-- Maps LlamaForCausalLM to the llama3 package.
    |   |   |   +-- Functions: register_all_models(); class: _LazyArch.
    |   |   +-- llama3/
    |   |       +-- __init__.py
    |   |       |   +-- Exports Llama3Model, Llama3Config, Llama3BatchProcessor, and llama_arch; no functions.
    |   |       +-- arch.py
    |   |       |   +-- Declarative architecture manifest; object: llama_arch = SupportedArchitecture(...); no functions.
    |   |       +-- model_config.py
    |   |       |   +-- Resolves 7B dimensions: layers=32, hidden=4096, heads=32, head_dim=4096/32=128 for a typical Llama 7B config.
    |   |       |   +-- Functions: create_rope_embedding(), Llama3Config.initialize(), finalize(), construct_kv_params(), calculate_attention_multiplier().
    |   |       +-- weight_adapters.py
    |   |       |   +-- Maps HF/GGUF names to MAX names, for example model.layers.0... -> layers.0....
    |   |       |   +-- Functions: convert_safetensor_state_dict(), convert_gguf_state_dict(), _convert_safetensor_with_model_config().
    |   |       +-- model.py
    |   |       |   +-- Loads weights, compiles the selected single-device/TP/DP graph, and executes model(*buffers) -> logits.
    |   |       |   +-- Functions: LlamaModelBase.load_model(), load_graphs(), graph_specs(), execute(), prepare_initial_token_inputs(), _load_weights().
    |   |       +-- llama3.py
    |   |       |   +-- Builds y=x+Attn(RMSNorm(x)); output=y+MLP(RMSNorm(y)).
    |   |       |   +-- Functions: Llama3.__init__(), trace(), input_tree(), input_types(), StackedMLP.__call__().
    |   |       +-- batch_processor.py
    |   |       |   +-- Packs [3-token, 2-token] rows into tokens=[t0,t1,t2,t3,t4], offsets=[0,3,5].
    |   |       |   +-- Functions: _stage_ragged_token_inputs(), prepare_initial_token_inputs(), get_symbolic_inputs(), process_outputs(), _make_inputs().
    |   |       +-- distributed_llama.py
    |   |       |   +-- Tensor parallelism: devices compute partitions of xW and combine their results.
    |   |       |   +-- Functions: DistributedLlama3.__init__(), trace(), input_tree(), input_types().
    |   |       +-- data_parallel_llama.py
    |   |       |   +-- Replicates the full model and assigns different requests to each replica.
    |   |       |   +-- Functions: DataParallelLlama.__call__(), trace(), input_types(), _call_flat(), create_model().
    |   |       +-- spec_adapters.py
    |   |       |   +-- Validates optional speculative-decoding pairings.
    |   |       |   +-- Functions: verify(), _verify().
    |   |       +-- BUILD.bazel
    |   |           +-- Declares sources and dependencies; build rule: modular_py_library(name="llama3").
    |   +-- lib/
    |   |   +-- config/
    |   |   |   +-- config.py
    |   |   |   |   +-- Resolves devices, context lengths, and runtime flags.
    |   |   |   |   +-- Functions: PipelineConfig.from_args(), configure_session(), _resolve_models_max_length(), _resolve_overlap_and_device_graph_capture().
    |   |   |   +-- model_config.py
    |   |   |       +-- Loads config.json and resolves checkpoint files.
    |   |   |       +-- Functions: MAXModelConfig.from_pipeline_args(), resolved_weight_paths(), architecture_name(), huggingface_config(), loader().
    |   |   +-- registry.py
    |   |   |   +-- Creates TextGenerationPipeline(config, Llama3Model, adapters, tokenizer, memory_plan).
    |   |   |   +-- Functions: PipelineRegistry.retrieve_factory(), retrieve_architecture(), retrieve_tokenizer(), architecture_for_config(), get_pipeline_for_task().
    |   |   +-- tokenizer.py
    |   |   |   +-- Converts text to token IDs and creates TextContext; exact IDs are tokenizer-specific.
    |   |   |   +-- Functions: TextTokenizer.encode(), decode(), new_context(), apply_chat_template(), create_eos_tracker(), max_tokens_to_generate().
    |   |   +-- graph_specs.py
    |   |   |   +-- Traces and compiles symbolic model and sampler graphs.
    |   |   |   +-- Functions: GraphSpec.trace(), weights_registry(), compile_and_init_graphs(), compile_roles_separately().
    |   |   +-- generate.py
    |   |   |   +-- Autoregressive rule: token[n+1] = sample(logits[n]).
    |   |   |   +-- Functions: GenerateIterator.from_contexts(), step(), close(), generate(), _build_contexts(), _check_batch_fits().
    |   |   +-- interfaces/
    |   |   |   +-- pipeline_model.py
    |   |   |   |   +-- Defines load -> compile -> prepare -> execute lifecycle.
    |   |   |   |   +-- Functions: PipelineModel.load_graphs(), prepare_initial_token_inputs(), _load_state_dict(), GraphPipelineModelWithKVCache.load_model().
    |   |   |   +-- batch_processor.py
    |   |   |       +-- Defines shared ragged-batch input/output contracts.
    |   |   |       +-- Functions: ragged_kv_symbolic_inputs(), ragged_kv_symbolic_input_tree(), ragged_batch_inputs(), process_ragged_kv_outputs().
    |   |   +-- pipeline_variants/
    |   |       +-- text_generation.py
    |   |       |   +-- Runs prepare -> forward -> sample -> context/KV update.
    |   |       |   +-- Functions: TextGenerationPipeline.prepare_batch(), execute(), _execute(), _launch_forward_pass(), load_sampler_graphs().
    |   |       +-- overlap_text_generation.py
    |   |           +-- Overlaps CPU preparation of batch k+1 with GPU execution of batch k.
    |   |           +-- Functions: OverlapTextGenerationPipeline.execute(), _execute(), _run_forward(), _sample_logits(), warmup_graph_capture().
    |   +-- sampling/
    |   |   +-- sampling.py
    |   |       +-- Computes p_i=softmax(z_i/T), followed by top-k, stochastic, or greedy selection.
    |   |       +-- Functions: TopKSampler.__call__(), token_sampler(), TokenSampler.sample_logits(), sample_logits_with_prev(), rejection_sampler().
    |   +-- kv_cache/
    |       +-- memory_planner.py
    |       |   +-- Estimates KV bytes approximately as 2*layers*tokens*kv_heads*head_dim*dtype_bytes.
    |       |   +-- Functions: estimate_weights_size(), estimate_activation_memory(), infer_max_batch_size(), with_activation_reservation().
    |       +-- paged_kv_cache/
    |           +-- cache_manager.py
    |               +-- Allocates and advances reusable KV pages.
    |               +-- Functions: PagedKVCacheManager.claim(), alloc(), runtime_inputs(), step(), release(), reset_prefix_cache().
    +-- nn/
        +-- transformer/
        |   +-- transformer.py
        |       +-- Executes decoder layers and logits=W_lm*RMSNorm(h_L).
        |       +-- Functions: Transformer.__call__(), _process_hidden_states(), TransformerBlock.__call__(), logits_postprocess(), forward_sequential_layers().
        +-- attention/
        |   +-- attention_with_rope.py
        |       +-- Attention=softmax(RoPE(Q)RoPE(K)^T/sqrt(128)+causal_mask)V for a 128-dimensional head.
        |       +-- Functions: AttentionWithRope.__call__(), shard(), materialize_kv_from_hidden(), wqkv(), qkv_weight_scale().
        +-- embedding.py
        |   +-- Maps N token IDs through E in R^(vocab x 4096) to X in R^(N x 4096).
        |   +-- Functions: Embedding.__call__(), VocabParallelEmbedding.__call__(), _per_device_call().
        +-- linear.py
        |   +-- Computes y=xW^T+b and the gated SwiGLU MLP.
        |   +-- Functions: Linear.__call__(), linear(), Linear.shard(), MLP.__call__(), GPTQLinear.__call__().
        +-- norm/
        |   +-- rms_norm.py
        |       +-- RMSNorm(x)=gamma*x/sqrt(mean(x^2)+epsilon).
        |       +-- Functions: RMSNorm.__call__(), RMSNorm.shard(), sharding_strategy().
        +-- rotary_embedding.py
            +-- Rotates paired Q/K coordinates with position-dependent sine/cosine angles.
            +-- Functions: RotaryEmbedding.freqs_cis(), freqs_cis_base(), compute_scale(), __call__(), Llama3RotaryEmbedding._compute_inv_freqs().
```
