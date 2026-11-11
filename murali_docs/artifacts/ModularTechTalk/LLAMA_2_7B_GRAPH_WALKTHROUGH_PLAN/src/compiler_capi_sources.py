"""Load offline upstream MLIR C API evidence at the repository's LLVM pin."""
import hashlib
import json
import re
import subprocess

EXTERNAL_PREFIX = 'external/llvm-project/'


def add_capi_sources(refs, files, repo, task, revision):
    manifest = json.loads((task/'references/llvm_capi/manifest.json').read_text())
    pin = subprocess.check_output(['git', 'show', f'{revision}:bazel/public-patches/llvm_source.bzl'], cwd=repo).decode()
    commit = re.search(r'LLVM_COMMIT = "([0-9a-f]+)"', pin)[1]
    if manifest['commit'] != commit or manifest['max_revision'] != revision:
        raise ValueError('MLIR C API snapshot does not match MAX’s pinned LLVM dependency')
    for item in manifest['files']:
        raw = (task/item['snapshot']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError(f'C API snapshot hash mismatch: {item["path"]}')
        target = task/'sources'/item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        files[item['path']] = dict(path=item['path'], text=raw.decode(), sha256=item['sha256'],
            source_origin='llvm-project', revision=commit, snapshot=item['snapshot'], upstream_url=item['url'])
    specs = [
        ('capi_interop', 'mlir/include/mlir-c/Bindings/Python/Interop.h', 76, 110, 'Python capsule names and _CAPIPtr / _CAPICreate contract'),
        ('capi_capsule', 'mlir/include/mlir-c/Bindings/Python/Interop.h', 326, 342, 'Operation capsule creation and named-pointer extraction'),
        ('capi_struct', 'mlir/include/mlir-c/IR.h', 31, 60, 'Opaque C API structs: MlirOperation holds void *ptr'),
        ('capi_wrap', 'mlir/include/mlir/CAPI/Wrap.h', 21, 38, 'C++ pointer ↔ C API handle wrap/unwrap helpers'),
        ('capi_operation_wrap', 'mlir/include/mlir/CAPI/IR.h', 22, 31, 'Generate wrap/unwrap for mlir::Operation and MlirOperation'),
        ('capi_context_get', 'mlir/lib/CAPI/IR/IR.cpp', 695, 697, 'mlirOperationGetContext returns the existing native context'),
        ('capi_factory', 'mlir/lib/Bindings/Python/IRCore.cpp', 1207, 1214, 'PyOperation::createFromCapsule: extract, validate, obtain context, wrap'),
        ('capi_wrapper_create', 'mlir/lib/Bindings/Python/IRCore.cpp', 917, 966, 'PyOperation construction, destruction and forOperation/createInstance'),
        ('capi_factory_bind', 'mlir/lib/Bindings/Python/IRCore.cpp', 4482, 4492, 'Bind Operation._CAPICreate to PyOperation::createFromCapsule'),
    ]
    for key, path, start, end, label in specs:
        file = EXTERNAL_PREFIX + path
        text = files[file]['text']
        code = ''.join(text.splitlines(keepends=True)[start-1:end])
        refs[key] = dict(id=key, file=file, start=start, end=end, line=start,
            label=label, symbol=label, language='cpp', alias='LLVM', revision=commit,
            source_origin='llvm-project', sha256=hashlib.sha256(code.encode()).hexdigest())
