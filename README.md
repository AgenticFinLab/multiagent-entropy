## MultiAgent-Entropy

A Python package to implement LLM-based multi-agent system from an entropy perspective.

### Overview

MultiAgent-Entropy is a framework for building and experimenting with multi-agent systems powered by Large Language Models (LLMs). The project focuses on understanding and optimizing agent interactions through entropy-based analysis, providing tools for evaluation, experimentation, and visualization of multi-agent behaviors.

### Features

- **Multi-Agent Architectures**: Support for various agent topologies including single, sequential, centralized, decentralized, full decentralized, debate, and hybrid modes
- **Entropy Analysis**: Comprehensive entropy calculation and analysis tools for understanding agent decision-making processes
- **Experiment Management**: Structured configuration system for running large-scale experiments with different models, datasets, and agent configurations
- **Evaluation Framework**: Built-in metrics calculation, result aggregation, and visualization capabilities
- **Flexible Configuration**: YAML-based configuration system for easy experiment setup and parameter tuning

### Installation

```bash
git clone https://github.com/AgenticFinLab/multiagent-entropy.git
cd multiagent-entropy
pip install -e .
```

### Project Structure

```
multiagent-entropy/
├── README.md                          ## Project overview and documentation
├── description.txt                    ## Project description
├── requirements.txt                   ## Python dependencies
├── setup.py                          ## Package setup configuration
├── configs/                          ## Configuration files
├── docs/                             ## Documentation
│   ├── evaluation-guidance.md        ## Evaluation framework documentation
│   ├── experiment-guidance.md        ## Experiment configuration guide
│   ├── Progress-record.md            ## Development progress log
│   └── reference.md                  ## Reference materials
├── examples/                         ## Example implementations
│   └── uTEST/                        ## Test examples
├── maep/                             ## Core package code
│   ├── __init__.py
│   ├── entropy_infer.py             ## Entropy inference utilities
│   ├── generic.py                   ## Generic agent implementation
│   └── language/                    ## Language model integrations
│       ├── single.py                ## Single agent mode
│       ├── sequential.py            ## Sequential agent mode
│       ├── centralized.py           ## Centralized agent mode
│       ├── decentralized.py         ## Decentralized agent mode
│       ├── full_decentralized.py    ## Full decentralized agent mode
│       ├── debate.py                ## Debate agent mode
│       └── hybrid.py                ## Hybrid agent mode
├── evaluation/                       ## Evaluation framework
│   ├── __init__.py
│   ├── data_loader.py              ## Data loading utilities
│   ├── entropy_analyzer.py         ## Entropy analysis tools
│   ├── evaluator.py               ## Evaluation logic
│   ├── experiment_analyzer.py     ## Experiment analysis
│   ├── metrics_calculator.py      ## Metrics calculation
│   ├── results_aggregator.py      ## Result aggregation
│   ├── utils.py                   ## Utility functions
│   └── results/                   ## Evaluation results
└── experiments/                     ## Experiment management
    ├── configs/                    ## Experiment configurations
    │   ├── base_config.yml
    │   ├── batch_example_qwen3_4b_gsm8k.yml
    │   ├── batch_example_qwen3_8b_gsm8k.yml
    │   ├── agent_specific/         ## Agent configurations
    │   ├── dataset_specific/        ## Dataset configurations
    │   ├── entropy_configs/         ## Entropy configurations
    │   ├── infer_configs/           ## Inference configurations
    │   └── model_specific/          ## Model configurations
    ├── configs_exp/                ## Generated experiment configs
    ├── data/                       ## Experiment data
    ├── logs/                       ## Experiment logs
    ├── results/                    ## Experiment results
    │   ├── aggregated/             ## Aggregated results
    │   └── raw/                    ## Raw results
    ├── scripts/                    ## Experiment scripts
    │   ├── run_experiment.py       ## Experiment runner
    │   ├── config_loader.py        ## Configuration loader
    │   └── result_aggregator.py   ## Result aggregator
    └── temp/                       ## Temporary files
```

### Quick Start

#### Running a Single Experiment

```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name qwen3-4b_gsm8k_single_agent \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/gsm8k.yml" \
  --entropy-config "experiments/configs/entropy_configs/standard.yml" \
  --infer-config "experiments/configs/infer_configs/cuda0.yml" \
  --agent-type "single"
```

#### Running Batch Experiments

```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --batch-config experiments/configs/batch_example_qwen3_4b_gsm8k.yml
```

#### Evaluating Results

```bash
cd /home/yuxuanzhao/multiagent-entropy
python evaluation/evaluator.py
```

### Agent Modes

The framework supports seven different agent architectures:

1. **Single Agent**: Linear agent topology with a single solver
2. **Sequential Agent**: Pipeline topology with planner -> solver -> critic -> judger
3. **Centralized Agent**: Two-layer topology with domain-specific agents and central orchestrator
4. **Decentralized Agent**: Sequential agents with loopback mechanism before final orchestration
5. **Full Decentralized Agent**: Sequential agents with full communication and loopback
6. **Debate Agent**: Multi-agent debate system with majority voting
7. **Hybrid Agent**: Two-layer topology with enhanced context sharing and feedback

### Documentation

- [Experiment Guidance](docs/experiment-guidance.md): Detailed guide on experiment configuration and execution
- [Evaluation Guidance](docs/evaluation-guidance.md): Comprehensive documentation on evaluation framework and metrics

### Supported Models

- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B

### Supported Datasets

- GSM8K
- AIME2024
- MMLU
- HumanEval

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Citation

If you use this project in your research, please cite:

```bibtex
@software{multiagent_entropy,
  title={MultiAgent-Entropy},
  author={AgenticFinLab},
  year={2026},
  url={https://github.com/AgenticFinLab/multiagent-entropy}
}
```

### Contact

For questions and support, please open an issue on GitHub.
