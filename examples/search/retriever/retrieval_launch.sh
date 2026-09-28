#!/usr/bin/env bash
set -euo pipefail

# Explicit ownership prevents detached services from outliving their training run.
: "${RETRIEVAL_OWNER_PID:?Set RETRIEVAL_OWNER_PID to the training launcher PID}"
: "${RETRIEVAL_JOB_ID:?Register the service before setting RETRIEVAL_JOB_ID}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$repo_root"
asset_data_dir="/home/shuixia/users/houguoqiang/code/datasets"
asset_weights_dir="/home/shuixia/users/houguoqiang/code/weights"
python_bin="${PYTHON_BIN:-/home/shuixia/miniconda3/envs/llm_ft_py310/bin/python}"
save_path="$asset_data_dir/searchR1"

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path="$asset_weights_dir/e5-base-v2"

exec "$python_bin" examples/search/retriever/lifecycle.py \
  --owner-pid "$RETRIEVAL_OWNER_PID" --job-id "$RETRIEVAL_JOB_ID" -- \
  "$python_bin" examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --port 8081 "$@"
