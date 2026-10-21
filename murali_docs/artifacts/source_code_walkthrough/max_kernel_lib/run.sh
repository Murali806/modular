#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
task_tmp="${script_dir}"

mkdir -p "${task_tmp}/output-user-root" "${task_tmp}/disk-cache"

echo "Repository revision: $(git -C "${repo_root}" rev-parse --short HEAD)"
echo "Demo directory: ${script_dir}"
echo "Bazel temporary data: ${task_tmp}"

cd "${repo_root}"
exec ./bazelw \
    --output_user_root="${task_tmp}/output-user-root" \
    test \
    --disk_cache="${task_tmp}/disk-cache" \
    --repo_contents_cache= \
    --config=prebuilt-mojo \
    --config=disable-lint \
    --aspects= \
    --output_groups=default \
    --enable_runfiles=true \
    --build_runfile_links=true \
    --remote_download_outputs=all \
    --test_output=all \
    //murali_docs/artifacts/source_code_walkthrough/max_kernel_lib:add_one_demo_test
