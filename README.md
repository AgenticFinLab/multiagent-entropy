
<div align="center">
<h1>On the Uncertainty of Large Language Model-Based Multi-Agent Systems</h1>


Yuxuan Zhao<sup>1,2</sup>,  Sijia Chen<sup>2</sup>†,  Ningxin Su<sup>2</sup>

<sup>1</sup> Yantai Research Institute of Harbin Engineering University <sup>2</sup> Hong Kong University of Science and Technology (Guangzhou)

<sup>†</sup> Corresponding Author  


<a href="https://arxiv.org/abs/2602.04234"><img src='https://img.shields.io/badge/arXiv-Multi%20Agent%20Uncertainty-red' alt='Paper PDF'>  </a>
</div>

### Overview

Multi-agent systems (MAS) have emerged as a prominent paradigm for leveraging large language models (LLMs) to tackle complex tasks. However, the mechanisms governing the effectiveness of MAS built upon publicly available LLMs, specifically the underlying rationales for their success or failure, remain largely unexplored. In this paper, we revisit MAS through the perspective of uncertainty, considering both intra- and inter-agent dynamics by investigating entropy transitions during problem-solving across various topologies and six benchmark tasks. By analyzing 245 features spanning token-, trajectory-, and round-level entropy, we counterintuitively find that a single agent outperforms MAS in approximately 43.3% of cases, and that uncertainty dynamics are largely determined during the first round of interaction. Furthermore, we provide three key observations: 1) Certainty Preference: reducing uncertainty at any stage for any agent is critical for guaranteeing correct solutions; 2) Base Uncertainty: base models with lower entropy during problem-solving directly benefit MAS performance; and 3) Task Awareness: entropy dynamics of MAS play varying roles across different tasks. Building on these insights, we introduce a simple yet effective algorithm, the Entropy Judger, to select solutions from MAS's pass@k results, leading to consistent accuracy improvements across all MAS configurations and tasks. 

---

### Installation

```bash
git clone https://github.com/AgenticFinLab/multiagent-entropy.git
cd multiagent-entropy
pip install -e .
```

---

### Project Structure

```
multiagent-entropy/
├── README.md                          ## Project overview and documentation
├── requirements.txt                   ## Python dependencies
├── setup.py                           ## Package setup configuration
├── maep/                              ## Core package: Multi-Agent Entropy Package
│   ├── entropy_infer.py               ## Entropy inference utilities
│   ├── generic.py                     ## Generic agent implementation
│   └── language/                      ## Seven MAS architectures
│       ├── single.py                  ## Single agent baseline
│       ├── sequential.py              ## Sequential pipeline
│       ├── centralized.py             ## Centralized orchestration
│       ├── decentralized.py           ## Decentralized with loopback
│       ├── full_decentralized.py      ## Full communication graph
│       ├── debate.py                  ## Majority voting
│       └── hybrid.py                  ## Enhanced context sharing
├── experiments/                       ## Experiment execution
│   ├── configs/                       ## Configuration files
│   │   ├── base_config.yml            ## Base configuration
│   │   ├── agent_specific/            ## Agent architecture configs
│   │   ├── dataset_specific/          ## Dataset configs (GSM8K, AIME, etc.)
│   │   └── model_specific/            ## Model configs (Qwen3 series)
│   ├── scripts/
│   │   ├── run_experiment.py          ## Main experiment runner
│   │   └── config_loader.py           ## Configuration loader
│   ├── results/
│   │   ├── raw/                       ## Raw experiment results
│   │   └── aggregated/                ## Aggregated results
│   └── data/                          ## Dataset storage
├── evaluation/                        ## Result evaluation and feature extraction
│   ├── evaluator.py                   ## Main evaluation entry
│   ├── aggregator.py                  ## Data aggregation
│   ├── entropy_statistic.py           ## Entropy statistics
│   ├── feature_enhancer.py            ## 245-feature extraction
│   └── results/                       ## Evaluation outputs
├── data_mining/                       ## Data mining analysis
│   ├── code/
│   │   ├── main.py                    ## Main entry for data mining
│   │   ├── data_mining_analyzer.py    ## Unified analyzer
│   │   ├── shap_analyzer.py           ## SHAP interpretability
│   │   └── run_experiments.py         ## Automated experiment runner
│   └── results/                       ## Analysis results
├── entropy_analysis/                  ## Auxiliary entropy visualization
│   ├── code/
│   │   ├── entropy_analyzer.py        ## Core analysis
│   │   ├── visualizer.py              ## Visualization
│   │   └── data_loader.py             ## Hierarchical data loading
│   └── visualizations/                ## Generated plots
└── docs/                              ## Documentation
    ├── experiment-guidance.md         ## Experiment guide
    ├── evaluation-guidance.md         ## Evaluation guide
    ├── data-mining-guidance.md        ## Data mining guide
    └── entropy-analysis-guide.md      ## Entropy analysis guide
```

---

### Quick Start

#### 1. Running Experiments

Run a single experiment:

```bash
python experiments/scripts/run_experiment.py \
  --experiment-name qwen3-4b_gsm8k_single_agent \
  --base-config experiments/configs/base_config.yml \
  --model-config experiments/configs/model_specific/qwen3-4b.yml \
  --dataset-config experiments/configs/dataset_specific/gsm8k.yml \
  --agent-type single
```

Run batch experiments:

```bash
python experiments/scripts/run_experiment.py \
  --batch-config experiments/configs/batch_example_qwen3_4b_gsm8k.yml
```

#### 2. Evaluating Results

Evaluate experiments and extract entropy features:

```bash
python -m evaluation.evaluator --dataset gsm8k --run-aggregator
```

#### 3. Data Mining Analysis

Analyze factors affecting sample-level correctness:

```bash
cd data_mining/code
python main.py --analysis-type all
```

#### 4. Entropy Visualization

Visualize entropy transitions across architectures:

```bash
cd entropy_analysis/code
python main.py --dataset gsm8k --multi-level
```

---

### Extensibility and Supported Configurations

#### 1. Supported MAS Architectures
- **Single**: Baseline with a single solver agent
- **Sequential**: Pipeline: planner → solver → critic → judger
- **Centralized**: Two-layer: domain agents + central orchestrator
- **Decentralized**: Sequential agents with loopback mechanism
- **Full Decentralized**: Fully connected communication among all agents
- **Debate**: Multi-agent debate with majority voting
- **Hybrid**: Two-layer with enhanced context sharing

#### 2. Supported Models and Datasets
- **Models**: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, LLaMA-3.1-8B-Instruct, LLaMA-3.2-3B-Instruct
- **Datasets**: GSM8K, AIME2024, AIME2025, MMLU, HumanEval, MATH-500  

#### 3. Adding New Components

**New MAS Architectures**  
To implement a custom architecture:  
1. Create a Python file in `maep/language/` (e.g., `my_architecture.py`).  
2. Define an architecture class following patterns in `centralized.py` or `hybrid.py`.  
3. Add agent configurations in `experiments/configs/agent_specific/my_architecture_agents.yml`.  
4. Register the new type in `experiments/scripts/run_experiment.py`.  

**Custom Models**  
To integrate a new model:  
1. Create a YAML config in `experiments/configs/model_specific/` (e.g., `my_model.yml`):  
   ```yaml
   lm_name: "path/to/your/model"
   inference_config:
     device: "cuda"
     torch_dtype: "float16"
   ```  
2. Specify it at runtime:  
   ```bash
   python experiments/scripts/run_experiment.py \
     --model-config experiments/configs/model_specific/my_model.yml
   ```

**Custom Datasets**  
To add a new dataset:  
1. Create a YAML config in `experiments/configs/dataset_specific/` (e.g., `my_dataset.yml`):  
   ```yaml
   data:
     data_name: "MyDataset"
     data_path: "experiments/data/MyDataset"
     split: "test"
     data_num: -1  # -1 for all samples
     batch_size: 1
   task_type: "math"  # math, code, or option
   generation_config:
     max_new_tokens: 2048
   ```  
2. Place data in `experiments/data/MyDataset/` using the expected format.  
3. Run with:  
   ```bash
   python experiments/scripts/run_experiment.py \
     --dataset-config experiments/configs/dataset_specific/my_dataset.yml
   ```

---
### Documentation

- [Experiment Guidance](docs/experiment-guidance.md): Experiment configuration and execution
- [Evaluation Guidance](docs/evaluation-guidance.md): Evaluation framework and feature extraction
- [Data Mining Guidance](docs/data-mining-guidance.md): Data mining and SHAP analysis
- [Entropy Analysis Guide](docs/entropy-analysis-guide.md): Entropy visualization and analysis

---
### Citation

If you find this work useful, please cite:

```bibtex
@article{multiagent-uncertainty,
  title={On the Uncertainty of Large Language Model-Based Multi-Agent Systems},
  author={Yuxuan Zhao, Sijia Chen, Ningxin Su},
  journal={arXiv preprint arXiv:2602.04234},
  year={2026},
}

```

