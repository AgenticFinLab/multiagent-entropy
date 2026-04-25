"""Resolve paths.yml into concrete per-plot kwargs.

All paths are relative to the project root via {root} substitution, so no
environment branching is needed — the same config works on every machine
that has the project checked out.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _substitute(value: Any, mapping: Dict[str, str]) -> Any:
    """Recursively expand {placeholder} tokens in strings."""
    if isinstance(value, str):
        try:
            return value.format_map(mapping)
        except KeyError:
            return value
    if isinstance(value, list):
        return [_substitute(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, mapping) for k, v in value.items()}
    return value


def load_config() -> dict:
    cfg_path = CONFIG_DIR / "paths.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config file found at {cfg_path}")
    cfg = _load_yaml(cfg_path)

    # Two-pass substitution: first {root}, then defaults references.
    mapping: Dict[str, str] = {"root": str(PROJECT_ROOT)}
    defaults_resolved = _substitute(cfg.get("defaults", {}), mapping)
    mapping.update({k: str(v) for k, v in defaults_resolved.items() if isinstance(v, str)})

    return {
        "defaults": defaults_resolved,
        "plots": _substitute(cfg.get("plots", {}), mapping),
    }


def get_plot_paths(plot_name: str) -> dict:
    """Return resolved kwargs for one plot, with `output_dir` filled in."""
    cfg = load_config()
    plots = cfg["plots"]
    if plot_name not in plots:
        raise KeyError(
            f"Plot '{plot_name}' not in config. Available: {sorted(plots.keys())}"
        )
    params = copy.deepcopy(plots[plot_name])
    if "output_dir" not in params:
        output_root = cfg["defaults"].get(
            "output_root", str(PROJECT_ROOT / "visualization" / "outputs")
        )
        params["output_dir"] = str(Path(output_root) / plot_name)
    return params
