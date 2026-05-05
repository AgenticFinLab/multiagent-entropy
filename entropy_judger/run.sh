#!/usr/bin/env bash
# Run K=20 repeated inferences for the Entropy Judger experiment.
#
# Results land in experiments/results_entropy_judger/raw/{dataset}/{model}/run{k}_*/
# which keeps them separate from the main experiments/results/ directory.
#
# Usage:
#   cd <repo_root>
#   bash entropy_judger/run.sh
#
# Scope control via environment variables:
#   MODELS="qwen3-4b" DATASETS="gsm8k" K=3 bash entropy_judger/run.sh
#   DATA_NUM="gsm8k:50,math500:100" bash entropy_judger/run.sh  (per-dataset)
#   DATA_NUM_DEFAULT=50 bash entropy_judger/run.sh               (all datasets)
#   CUDA_VISIBLE_DEVICES=2 bash entropy_judger/run.sh            (specify GPU)
#   MAX_RETRIES=5 RETRY_DELAY=60 bash entropy_judger/run.sh      (retry config)

set -e

MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-30}"  # seconds to wait between retries

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

MODELS="${MODELS:-qwen3-4b qwen3-8b llama-3.1-8b-instruct llama-3.2-3b-instruct qwen3-0.6b}"
DATASETS="${DATASETS:-gsm8k math500 humaneval mmlu aime2024 aime2025}"
ARCHITECTURES="${ARCHITECTURES:-single sequential centralized debate hybrid}"
K="${K:-20}"
# DATA_NUM: per-dataset sample counts, e.g. "gsm8k:50,math500:100"
# DATA_NUM_DEFAULT: fallback count for datasets not listed in DATA_NUM (unset = use dataset config)
DATA_NUM="${DATA_NUM:-}"
DATA_NUM_DEFAULT="${DATA_NUM_DEFAULT:-}"

# Build associative array from DATA_NUM string
declare -A DATASET_NUM
if [ -n "$DATA_NUM" ]; then
  IFS=',' read -ra PAIRS <<< "$DATA_NUM"
  for pair in "${PAIRS[@]}"; do
    key="${pair%%:*}"
    val="${pair##*:}"
    DATASET_NUM["$key"]="$val"
  done
fi

# Return the --data-num flag for a given dataset, or empty string
get_data_num_flag() {
  local ds="$1"
  if [ -n "${DATASET_NUM[$ds]+_}" ]; then
    echo "--data-num ${DATASET_NUM[$ds]}"
  elif [ -n "$DATA_NUM_DEFAULT" ]; then
    echo "--data-num $DATA_NUM_DEFAULT"
  else
    echo ""
  fi
}

BASE_CFG="experiments/configs/base_config.yml"
MODEL_DIR="experiments/configs/model_specific"
DATASET_DIR="experiments/configs/dataset_specific"
RUNNER="entropy_judger/run_single.py"

# Run a single experiment with automatic retry on failure
run_with_retry() {
  local exp_name="$1"; shift
  local attempt=1
  while [ "$attempt" -le "$MAX_RETRIES" ]; do
    # shellcheck disable=SC2086
    if python "$RUNNER" "$@"; then
      return 0
    fi
    echo "  [WARN] $exp_name failed (attempt $attempt/$MAX_RETRIES), retrying in ${RETRY_DELAY}s ..."
    sleep "$RETRY_DELAY"
    attempt=$((attempt + 1))
  done
  echo "  [ERROR] $exp_name failed after $MAX_RETRIES attempts, skipping."
  return 0  # don't abort the whole script
}

for k in $(seq 1 "$K"); do
  echo "===== Repeat $k / $K ====="
  for model in $MODELS; do
    for dataset in $DATASETS; do
      # HumanEval has no Debate architecture
      archs="$ARCHITECTURES"
      if [ "$dataset" = "humaneval" ]; then
        archs="${ARCHITECTURES/debate/}"
      fi
      for arch in $archs; do
        [ -z "$arch" ] && continue
        exp_name="run${k}_${dataset}_${model}_${arch}"
        data_num_flag=$(get_data_num_flag "$dataset")
        echo "  Running: $exp_name${data_num_flag:+ ($data_num_flag)}"
        run_with_retry "$exp_name" \
          -b "$BASE_CFG" \
          -m "${MODEL_DIR}/${model}.yml" \
          -d "${DATASET_DIR}/${dataset}.yml" \
          -n "$exp_name" \
          --agent-type "$arch" \
          $data_num_flag
      done
    done
  done
done

echo "All $K repeats complete."
