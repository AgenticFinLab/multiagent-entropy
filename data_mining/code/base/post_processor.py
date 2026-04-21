"""BasePostProcessor — shared iteration over (model × dataset × architecture)
trees produced by the analyzers.

Kept minimal on purpose: it's a simple iterator contract so aggregator.py,
visualizer.py, and summarizer.py can share the walk without each reimplementing
``for model in ...: for dataset in ...: for architecture in ...:``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional


@dataclass
class ExperimentContext:
    """One leaf of the experiment tree."""

    model: str
    dataset: str
    architecture: str
    path: Path


class BasePostProcessor:
    """Iterate results-tree leaves in a canonical order."""

    def iterate_experiments(
        self,
        results_dir: Path,
        callback: Optional[Callable[[ExperimentContext], None]] = None,
    ) -> Iterator[ExperimentContext]:
        """Yield each ``<model>/<dataset>/<architecture>/`` leaf under ``results_dir``.

        If ``callback`` is given, it is invoked once per leaf and the iterator
        still yields each context so callers can either consume the iterator
        or rely on side effects.
        """
        results_dir = Path(results_dir)
        if not results_dir.exists():
            return
        for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
            for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                for arch_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                    ctx = ExperimentContext(
                        model=model_dir.name,
                        dataset=dataset_dir.name,
                        architecture=arch_dir.name,
                        path=arch_dir,
                    )
                    if callback is not None:
                        callback(ctx)
                    yield ctx
