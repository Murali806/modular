#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

bazel_args=(
  --config=prebuilt-mojo
  --disk_cache=/local/mnt/workspace/.murali_bazel/disk
  --repository_cache=/local/mnt/workspace/.murali_bazel/repo
)

./bazelw --output_user_root=/local/mnt/workspace/.murali_bazel/user run \
  "${bazel_args[@]}" \
  //max/python/max/_entrypoints:pipelines.venv >/dev/null

venv="$repo_root/.max+python+max+_entrypoints+pipelines.venv"
runfiles="$(realpath \
  bazel-bin/max/python/max/_entrypoints/pipelines.venv.runfiles)"
platlib="$(find "$runfiles" -maxdepth 1 -type d \
  -name '+rebuild_wheel+module_platlib_*' -print -quit)"

if [[ -z "$platlib" ]]; then
  echo "Could not locate the MAX native-extension runfiles." >&2
  exit 1
fi

export PYTHONPATH="$platlib${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$venv/bin/python" \
  "$repo_root/murali_docs/labs/kv_cache/kv_cache_trace.py" "$@"
