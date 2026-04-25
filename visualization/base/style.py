"""Global ICML plotting style and shared color palettes."""

import matplotlib.pyplot as plt


ARCH_ORDER = ["centralized", "debate", "hybrid", "sequential", "single"]
ARCH_ORDER_WITH_BASE = ["base"] + ARCH_ORDER

ARCH_COLORS = {
    "base": "#D3D3D3",
    "centralized": "#D73027",
    "debate": "#FC8D59",
    "hybrid": "#FEE090",
    "sequential": "#4575B4",
    "single": "#91BFD8",
}

FEATURE_COLORS = {
    "sample_round_1_q3_agent_max_entropy": "#91BFD8",
    "sample_round_1_max_agent_max_entropy": "#FEE090",
    "sample_round_1_max_agent_std_entropy": "#FEE090",
}


def apply_icml_style(base_font_size: int = 14) -> None:
    """Apply ICML-style matplotlib rcParams.

    Call once per script (BaseVisualizer.__init__ does this for you).
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = base_font_size
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.labelsize"] = base_font_size - 1
    plt.rcParams["ytick.labelsize"] = base_font_size - 1
    plt.rcParams["axes.labelsize"] = base_font_size + 1
    plt.rcParams["legend.title_fontsize"] = base_font_size
    plt.rcParams["legend.fontsize"] = base_font_size - 2
