"""Source catalog for the offline Llama 2 reader. No network or MAX required."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
REPO = TASK.parents[3]
OUT = TASK / "link_htmls" / "llama_2_7b"
REVISION = "610f7111836d9a950d45ba52ec9b3e0ed589fe3f"

PATHS = {
    "arch": "max/python/max/pipelines/architectures/llama3/arch.py",
    "config": "max/python/max/pipelines/architectures/llama3/model_config.py",
    "model": "max/python/max/pipelines/architectures/llama3/model.py",
    "llama": "max/python/max/pipelines/architectures/llama3/llama3.py",
    "adapter": "max/python/max/pipelines/architectures/llama3/weight_adapters.py",
    "batch": "max/python/max/pipelines/architectures/llama3/batch_processor.py",
    "pipeline": "max/python/max/pipelines/lib/pipeline_variants/text_generation.py",
    "response_utils": "max/python/max/pipelines/lib/pipeline_variants/utils.py",
    "scheduler": "max/python/max/serve/scheduler/text_generation_scheduler.py",
    "batch_constructor": "max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py",
    "spec": "max/python/max/pipelines/lib/graph_specs.py",
    "pipeline_config": "max/python/max/pipelines/lib/config/config.py",
    "model_config": "max/python/max/pipelines/lib/config/model_config.py",
    "hf": "max/python/max/pipelines/weights/hf_utils.py",
    "batch_types": "max/python/max/pipelines/lib/interfaces/batch_processor.py",
    "planner": "max/python/max/pipelines/kv_cache/memory_planner.py",
    "sampler": "max/python/max/pipelines/sampling/sampling.py",
    "sampling_processor": "max/python/max/pipelines/sampling/sampling_logits_processor.py",
    "transformer": "max/python/max/nn/transformer/transformer.py",
    "attention": "max/python/max/nn/attention/attention_with_rope.py",
    "linear": "max/python/max/nn/linear.py",
    "rope": "max/python/max/nn/rotary_embedding.py",
    "norm": "max/python/max/nn/norm/rms_norm.py",
    "kv": "max/python/max/nn/kv_cache/cache_params.py",
    "attn_metadata": "max/python/max/nn/kv_cache/utils.py",
    "kernels": "max/python/max/nn/kernels.py",
    "graph": "max/python/max/graph/graph.py",
    "types": "max/python/max/graph/type.py",
    "tensor_shape": "Support/include/Support/ML/TensorShape.h",
    "mlir": "max/python/max/_mlir_context.py",
    "mlir_native": "external/llvm-project/mlir/lib/IR/MLIRContext.cpp",
    "init": "Init/lib/Init.cpp",
    "engine": "max/python/max/engine/api.py",
    "build": "max/python/max/_entrypoints/BUILD.bazel",
    "llama_build": "max/python/max/pipelines/architectures/llama3/BUILD.bazel",
    "builtin": "max/kernels/src/graph_compiler/builtin_kernels/kernels.mojo",
    "builtin_attention": "max/kernels/src/graph_compiler/builtin_kernels/attention.mojo",
    "mha": "max/kernels/src/nn/attention/gpu/mha.mojo",
    "ragged_kernel": "max/kernels/src/nn/kv_cache_ragged.mojo",
    "kv_kernel": "max/kernels/src/nn/kv_cache.mojo",
    "rope_kernel": "max/kernels/src/nn/rope_split_store.mojo",
    "kv_types": "max/kernels/src/kv_cache/types.mojo",
    "frontend": "GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp",
    "compiler": "GraphCompiler/lib/GraphCompiler.cpp",
    "mo_map": "GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp",
    "map_mogg": "GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp",
    "map_state": "GraphCompiler/lib/Interfaces/MAPToMOGGState.cpp",
    "fusion": "GraphCompiler/lib/MAPDialect/Transforms/FuseKernels.cpp",
    "mutable_fusion": "GraphCompiler/lib/MAPDialect/Transforms/FuseMutableLoads.cpp",
    "conform": "GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp",
    "mogg_mgp": "GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp",
    "lower_primitives": "GraphCompiler/lib/Conversion/MOGGToMGP/LowerPrimitives.cpp",
    "memory": "GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp",
    "emit": "GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp",
    "emit_state": "GraphCompiler/lib/Interfaces/EmitMojoState.cpp",
    "jit": "GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp",
    "object_compiler": "Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp",
    "llvm_pipeline": "Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp",
    "kgen_passes": "Mojo/include/Mojo/KGENPasses.td",
    "nvptx": "Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp",
    "host_loader": "GraphCompiler/lib/Support/CompiledKernelsContext.cpp",
    "primitive": "GraphCompiler/lib/MGPDialect/Primitives/MGPPrimitives.cpp",
    "mef": "GraphCompiler/lib/MEFModel/MEFModel.cpp",
    "invokable": "MLRT/lib/GraphRT/MEFExecutor/MEFInvokable.cpp",
    "async": "AsyncRT/include/AsyncRT/Runtime/AsyncValue.h",
    "queue": "AsyncRT/lib/Runtime/ThreadPoolWorkQueue.cpp",
    "cpu": "AsyncRT/lib/Runtime/CPUDevice.cpp",
    "cpu_header": "AsyncRT/include/AsyncRT/Runtime/CPUDevice.h",
    "virtual": "AsyncRT/include/AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h",
    "device_context": "MLRT/lib/Driver/DeviceContext/DeviceContext.cpp",
    "cuda": "MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.cpp",
    "session_native": "max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp",
    "engine_context": "max/lib/max_license/API/c/EngineContext.cpp",
    "execution": "max/lib/max_license/API/c/ModelExecution.cpp",
    "device_native": "max/python/max/_core/max_license/lib/Driver/Device.cpp",
    "buffer_native": "max/python/max/_core/max_license/lib/Driver/Buffer.cpp",
}


def embedded_sources():
    result = {}
    for filename, key in [
        ("max_example_04_fused_attention_online_softmax.html", "attention-data"),
        ("max_example_04_asyncrt_runtime_deep_dive.html", "virtual-device-data"),
    ]:
        html = (TASK / "link_htmls" / filename).read_text()
        match = re.search(r'<script\b[^>]*id=[\"\x27]' + key + r'[\"\x27][^>]*>(.*?)</script>', html, re.S)
        data = json.loads(match[1])
        assert data["revision"] == REVISION
        for path, value in data["files"].items():
            result[path] = value["text"] if isinstance(value, dict) else value
    return result


def load_sources():
    embedded = embedded_sources()
    result = {}
    for alias, path in PATHS.items():
        local, snapshot = REPO / path, TASK / "sources" / path
        if local.is_file():
            text, origin = local.read_text(), "repository"
        elif snapshot.is_file():
            text, origin = snapshot.read_text(), "pinned snapshot"
        elif path in embedded:
            text, origin = embedded[path], "Example 04 embedded snapshot"
        else:
            candidates = [p for p in embedded if p.endswith(path.rsplit("/", 1)[-1])]
            raise FileNotFoundError(f"{alias}: {path}; possible embedded paths: {candidates}")
        result[alias] = dict(path=path, text=text, origin=origin, revision=REVISION,
                             sha256=hashlib.sha256(text.encode()).hexdigest())
    return result


def resolve(sources, alias, needle, occurrence=1):
    """Resolve literal anchors at build time; fail closed when source drifts."""
    lines = sources[alias]["text"].splitlines()
    hits = [i + 1 for i, line in enumerate(lines) if needle in line]
    if len(hits) < occurrence:
        raise ValueError(f"Missing source anchor {alias}: {needle!r} ({occurrence})")
    return dict(file=alias, line=hits[occurrence - 1], label=needle.strip())
