"""Offline upstream source pinned to the LLVM commit selected by this checkout."""
from pathlib import Path
import hashlib
import json

LLVM_REVISION = '6941494e2ae550d753728ba13454311b344d82cf'
LLVM_FILES = {
    'llvm-object': 'llvm/lib/ExecutionEngine/Orc/ObjectLinkingLayer.cpp',
    'llvm-link': 'llvm/lib/ExecutionEngine/JITLink/JITLinkGeneric.cpp',
    'llvm-link-h': 'llvm/lib/ExecutionEngine/JITLink/JITLinkGeneric.h',
    'llvm-layout': 'llvm/lib/ExecutionEngine/JITLink/JITLinkMemoryManager.cpp',
    'llvm-mapper': 'llvm/lib/ExecutionEngine/Orc/MapperJITLinkMemoryManager.cpp',
    'llvm-memory': 'llvm/lib/ExecutionEngine/Orc/MemoryMapper.cpp',
    'llvm-link-layer': 'llvm/lib/ExecutionEngine/Orc/LinkGraphLinkingLayer.cpp',
    'llvm-core': 'llvm/include/llvm/ExecutionEngine/Orc/Core.h',
    'llvm-x86': 'llvm/include/llvm/ExecutionEngine/JITLink/x86_64.h',
}
LLVM_SOURCES = {key: f'external/llvm-project/{LLVM_REVISION}/{path}'
                for key, path in LLVM_FILES.items()}

RESEARCH_LINKS = [
    ('LLVM ORCv2', 'https://llvm.org/docs/ORCv2.html',
     'Separate registration, materialization, address resolution and readiness.'),
    ('LLVM JITLink', 'https://llvm.org/docs/JITLink.html',
     'Draw object sections, a LinkGraph, working memory and relocation fixups.'),
    ('CUDA module management',
     'https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MODULE.html',
     'Keep GPU image bytes, module handles and function handles distinct.'),
    ('CUDA launch API',
     'https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html',
     'Show pointers to argument-value slots and submission before completion.'),
    ('HIP module API',
     'https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/module_management.html',
     'Use the actual HIP module/function/launch APIs on the AMD route.'),
]


def add_llvm_sources(task: Path, files: dict):
    root = task / 'references' / 'llvm_orc' / LLVM_REVISION
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['revision'] == LLVM_REVISION
    for key, path in LLVM_FILES.items():
        raw = (root / path).read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        assert checksum == manifest['files'][path]['sha256'], path
        logical = LLVM_SOURCES[key]
        destination = task / 'sources' / logical
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        files[logical] = dict(text=raw.decode(), sha256=checksum,
                              revision=LLVM_REVISION, origin='LLVM upstream',
                              snapshot=str((root / path).relative_to(task)),
                              url=manifest['files'][path]['url'])
