#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

base_url="${BASE_URL:-http://127.0.0.1:18000}"
metrics_url="${METRICS_URL:-http://127.0.0.1:18001/metrics}"
model="${MODEL:-modularai/SmolLM-135M-Instruct-FP32}"
out_dir="${OUT_DIR:-/tmp/murali_max_cpu_benchmark}"
mkdir -p "$out_dir/logs"

curl --fail --silent --show-error "$base_url/v1/health" >/dev/null

HF_HOME="${HF_HOME:-/local/mnt/workspace/.murali_hf}" \
  ./bazelw --output_user_root=/local/mnt/workspace/.murali_bazel/user run \
  --config=prebuilt-mojo \
  --disk_cache=/local/mnt/workspace/.murali_bazel/disk \
  --repository_cache=/local/mnt/workspace/.murali_bazel/repo \
  //max/python/max/_entrypoints:pipelines -- benchmark \
  --backend modular \
  --base-url "$base_url" \
  --model "$model" \
  --tokenizer "$model" \
  --tokenizer-local-files-only \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 32 \
  --random-output-len 8 \
  --max-output-len 8 \
  --num-prompts 40 \
  --max-concurrency 1,2,4 \
  --request-rate inf \
  --skip-first-n-requests 0 \
  --skip-last-n-requests 0 \
  --no-collect-gpu-stats \
  --disable-tqdm \
  --metrics-urls.orchestrator="$metrics_url" \
  --result-filename "$out_dir/results.json" \
  --log-dir "$out_dir/logs" \
  --metadata device=cpu \
  --metadata purpose=semantics_only

python3 murali_docs/labs/performance/summarize_benchmark.py \
  "$out_dir"/logs/results-*-median.json >"$out_dir/summary.json"

cat "$out_dir/summary.json"
