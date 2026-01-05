CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_experiment.py \
  --experiment-name qwen3-4b_gsm8k_single_agent \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/gsm8k.yml" \
  --agent-type "single"

CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_gsm8k_sequential_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/gsm8k.yml" \
  --agent-type "sequential"

CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_experiment.py \
  --experiment-name qwen3-4b_gsm8k_centralized_agent \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/gsm8k.yml" \
  --agent-type "centralized"

CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_experiment.py \
  --experiment-name qwen3-4b_aime2024_single_agent \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --agent-type "single"

CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2024_sequential_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --agent-type "sequential"

CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2024_centralized_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --agent-type "centralized"

CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_experiment.py \
  --experiment-name qwen3-4b_math500_single_agent \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/math500.yml" \
  --agent-type "single"

CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_math500_sequential_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/math500.yml" \
  --agent-type "sequential"

CUDA_VISIBLE_DEVICES=0,1 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_math500_centralized_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/math500.yml" \
  --agent-type "centralized"