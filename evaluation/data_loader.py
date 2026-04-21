"""Standard data loader for multi-agent experiment results.

Targets `experiments/results/raw` (and `experiments/results_finagent/raw`).
All loading logic lives in `BaseDataLoader`; this class only declares the
storage layout via `_init_paths()`.
"""

from .base.data_loader import BaseDataLoader


class DataLoader(BaseDataLoader):
    """Loader for the standard experiment results tree."""

    def _init_paths(self) -> None:
        self.results_path = self.base_path / "experiments" / "results" / "raw"
        self.configs_path = self.base_path / "experiments" / "configs_exp"
        self.data_path = self.base_path / "experiments" / "data"
        self.results_finagent_path = (
            self.base_path / "experiments" / "results_finagent" / "raw"
        )
