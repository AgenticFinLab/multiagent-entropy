CUDA_VISIBLE_DEVICES=2,3 python experiments/scripts/run_experiment.py \
  --experiment-name qwen3-4b_gsm8k_debate_agent \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/gsm8k.yml" \
  --entropy-config "experiments/configs/entropy_configs/standard.yml" \
  --infer-config "experiments/configs/infer_configs/cuda_auto.yml" \
  --agent-type "debate"

CUDA_VISIBLE_DEVICES=2,3 python experiments/scripts/run_experiment.py \
  --experiment-name qwen3-4b_gsm8k_hybrid_agent \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/gsm8k.yml" \
  --entropy-config "experiments/configs/entropy_configs/standard.yml" \
  --infer-config "experiments/configs/infer_configs/cuda_auto.yml" \
  --agent-type "hybrid"

CUDA_VISIBLE_DEVICES=2,3 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2024_debate_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --entropy-config "experiments/configs/entropy_configs/standard.yml" \
  --infer-config "experiments/configs/infer_configs/cuda_auto.yml" \
  --agent-type "debate"

CUDA_VISIBLE_DEVICES=2,3 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2024_hybrid_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --entropy-config "experiments/configs/entropy_configs/standard.yml" \
  --infer-config "experiments/configs/infer_configs/cuda_auto.yml" \
  --agent-type "hybrid"

CUDA_VISIBLE_DEVICES=2,3 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_math500_debate_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/math500.yml" \
  --entropy-config "experiments/configs/entropy_configs/standard.yml" \
  --infer-config "experiments/configs/infer_configs/cuda_auto.yml" \
  --agent-type "debate"

CUDA_VISIBLE_DEVICES=2,3 python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_math500_hybrid_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/math500.yml" \
  --entropy-config "experiments/configs/entropy_configs/standard.yml" \
  --infer-config "experiments/configs/infer_configs/cuda_auto.yml" \
  --agent-type "hybrid"