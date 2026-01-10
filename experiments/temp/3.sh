CUDA_VISIBLE_DEVICES=3 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2025_single_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "single"

CUDA_VISIBLE_DEVICES=3 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2025_sequential_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "sequential"


CUDA_VISIBLE_DEVICES=3 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2025_centralized_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "centralized"

CUDA_VISIBLE_DEVICES=3 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2025_debate_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "debate"

CUDA_VISIBLE_DEVICES=3 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2025_hybrid_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "hybrid"

