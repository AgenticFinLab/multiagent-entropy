"""I/O helpers for data-mining analyzers.

- ``OutputManager`` — encapsulates the ``determine_output_directory`` +
  ``generate_filter_suffix`` + ``create_output_directory`` chain that lives
  in utils.py and was previously called separately in each analyzer.
- ``save_plot`` — replaces the repeated ``plt.tight_layout(); plt.savefig(...,
  dpi=300, bbox_inches="tight"); plt.close()`` boilerplate.
- ``load_dataset_csv`` — thin wrapper over ``utils.load_data_from_path`` for
  uniform import surface inside the base subpackage.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

# Re-use the existing helpers in utils.py to avoid behavior drift.
from utils import (
    load_data_from_path,
    determine_output_directory,
    generate_filter_suffix,
    create_output_directory,
)

from .constants import PLOT_DEFAULTS


class OutputManager:
    """Build and ensure the canonical output directory for an analyzer run."""

    def __init__(
        self,
        base_output_dir: str,
        analyzer_type: str = "",
        target_dataset: Optional[str] = None,
        dataset_type: str = "standard",
        model_names: Optional[List[str]] = None,
        architectures: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        with_filter_suffix: bool = False,
    ):
        self.base_output_dir = base_output_dir
        self.analyzer_type = analyzer_type
        self.target_dataset = target_dataset
        self.dataset_type = dataset_type
        self.model_names = model_names
        self.architectures = architectures
        self.datasets = datasets
        self.with_filter_suffix = with_filter_suffix

    def resolve(self) -> Path:
        """Compute the final output directory path; create it on disk."""
        path = determine_output_directory(
            self.base_output_dir,
            self.target_dataset,
            self.analyzer_type,
            self.dataset_type,
        )
        if self.with_filter_suffix:
            suffix = generate_filter_suffix(
                model_names=self.model_names,
                architectures=self.architectures,
                datasets=self.datasets,
            )
            if suffix:
                path = f"{path}/{suffix}"
        out = Path(path)
        create_output_directory(out)
        return out


def save_plot(save_path, dpi: Optional[int] = None, **savefig_kwargs) -> None:
    """Standardized figure save: tight_layout + savefig + close.

    Mirrors the per-analyzer pattern exactly, including ``dpi=300`` and
    ``bbox_inches="tight"`` defaults.
    """
    if dpi is None:
        dpi = PLOT_DEFAULTS["dpi"]
    savefig_kwargs.setdefault("bbox_inches", PLOT_DEFAULTS["bbox_inches"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, **savefig_kwargs)
    plt.close()


def load_dataset_csv(data_path) -> pd.DataFrame:
    """Load a merged-dataset CSV via the canonical loader in utils.py."""
    return load_data_from_path(Path(data_path))
