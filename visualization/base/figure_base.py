"""BaseVisualizer: shared init / save plumbing for all figure scripts."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from visualization.base.style import apply_icml_style


class BaseVisualizer:
    """Common base for all figure scripts.

    Subclasses are expected to:
      - implement compose() which builds and saves the main figure
      - optionally implement save_individual_subplots()
    """

    def __init__(
        self,
        output_dir: Path | str,
        base_font_size: int = 14,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.individual_dir = self.output_dir / "individual_subplots"

        apply_icml_style(base_font_size=base_font_size)

    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        dpi: int = 1200,
        fmt: Optional[str] = None,
    ) -> Path:
        """Save a figure under output_dir. Format inferred from extension if fmt is None."""
        out_path = self.output_dir / filename
        fmt = fmt or out_path.suffix.lstrip(".") or "pdf"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=fmt)
        print(f"Saved: {out_path}")
        return out_path

    def save_subplot(
        self,
        fig: plt.Figure,
        subplot_name: str,
        dpi: int = 300,
        fmt: str = "pdf",
    ) -> Path:
        """Save a per-subplot figure under output_dir/individual_subplots/."""
        self.individual_dir.mkdir(parents=True, exist_ok=True)
        safe = subplot_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        out_path = self.individual_dir / f"{safe}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=fmt)
        print(f"Saved subplot: {out_path}")
        return out_path
