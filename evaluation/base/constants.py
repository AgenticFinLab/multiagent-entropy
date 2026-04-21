"""Shared constants for the evaluation package.

Single source of truth for the dataset list and dataset-to-task-type mapping
that was previously duplicated across `evaluator.py`, `experiment_analyzer.py`,
and `temperature_ablation_evaluator.py`.
"""

DATASETS = [
    "gsm8k",
    "humaneval",
    "mmlu",
    "math500",
    "aime2024_16384",
    "aime2025_16384",
    "finagent",
]


DATASET_TASK_MAP = {
    "humaneval": "code",
    "mmlu": "option",
    "gsm8k": "math",
    "math500": "math",
    "aime2024_16384": "math",
    "aime2025_16384": "math",
    "aime2024_8192": "math",
    "aime2025_8192": "math",
    "finagent": "finance",
}


def infer_task_type(dataset: str, explicit: str = "auto") -> str:
    """Resolve task type from a dataset name.

    Honors an explicit non-auto override; otherwise falls back to the dataset
    map. Defaults to "math" for unknown datasets, matching the legacy behavior.
    The "math" sentinel is also treated as auto because the legacy code paths
    used it as the default CLI value.
    """
    if explicit not in ("auto", "math"):
        return explicit
    return DATASET_TASK_MAP.get(dataset.lower(), "math")
