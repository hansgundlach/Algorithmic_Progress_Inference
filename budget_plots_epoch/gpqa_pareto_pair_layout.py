"""
Shared layout for GPQA Pareto figures that are placed side-by-side in LaTeX.

Problem: `tight_layout()` and `bbox_inches="tight"` resize the axes differently when
titles have different heights (e.g. one-line vs two-line), so the data box looks mismatched.

Fix: use identical figure size and **fixed** `subplots_adjust` margins so the axes
position is the same in figure coordinates; save the full figure without tight cropping.
"""

from __future__ import annotations

from pathlib import Path

# Match both scripts (ICML-style side-by-side pair)
PAIR_FIGSIZE_INCHES = (7.0, 6.0)
PAIR_FIG_DPI = 150

# Title / axis fonts (keep in sync across pair)
TITLE_FONTSIZE = 16
TITLE_PAD = 12
TICK_LABELSIZE = 12
AXIS_LABELSIZE = 14
LEGEND_FONTSIZE = 11

# Shared axis limits — both plots show the same data range
PAIR_XMIN = 0.001 * 0.5  # 0.0005
PAIR_XMAX = 100
PAIR_YMIN = 0.08
PAIR_YMAX = 0.90

# Fixed axes position in figure coordinates (same for every plot in the pair).
# Top margin is large enough for a **two-line** title; single-line titles get extra
# whitespace above the axes but the x/y box matches the MoE/Dense plot.
PAIR_SUBPLOT_ADJUST = dict(left=0.12, right=0.97, bottom=0.11, top=0.84)


def apply_pair_layout(fig) -> None:
    """Replace tight_layout() with fixed margins for identical axes boxes."""
    fig.subplots_adjust(**PAIR_SUBPLOT_ADJUST)


def savefig_pair(fig, path: str, dpi: int = 300, **kwargs) -> None:
    """Save full figure (no bbox_inches='tight') so dimensions match across the pair."""
    kwargs.setdefault("facecolor", "white")
    kwargs.setdefault("edgecolor", "none")
    fig.savefig(path, dpi=dpi, **kwargs)


def _print_usage() -> None:
    print(
        "gpqa_pareto_pair_layout.py — library only (layout constants + helpers).\n"
        ">>> This command does NOT create any file in ../figures/. Nothing is wrong.\n\n"
        "To actually write PNGs into figures/, run from this directory:\n"
        "  python gpqa_pareto_names.py\n"
        "      -> ../figures/gpqa_pareto_with_names.png (+ .pdf)\n"
        "  python moe_dense_pareto_frontier.py --no-show\n"
        "      -> ../figures/moe_dense_pareto_gpqa_d.png\n\n"
        "Optional: python gpqa_pareto_pair_layout.py --demo\n"
        "      -> _pair_layout_demo.png here in budget_plots_epoch/ only (not figures/)\n"
    )


def _demo() -> None:
    """Minimal figure to verify matplotlib and apply_pair_layout."""
    import matplotlib

    matplotlib.use("Agg")  # no GUI needed
    import matplotlib.pyplot as plt

    print(f"matplotlib {matplotlib.__version__}")
    fig, ax = plt.subplots(figsize=PAIR_FIGSIZE_INCHES, dpi=PAIR_FIG_DPI)
    ax.plot([0, 1], [0, 1])
    ax.set_title("pair layout demo (shared axes box)\nsecond line")
    apply_pair_layout(fig)
    out = Path(__file__).resolve().parent / "_pair_layout_demo.png"
    savefig_pair(fig, str(out), dpi=150)
    plt.close(fig)
    print(f"Wrote demo: {out}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="GPQA Pareto pair layout helpers (library).")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Draw a tiny test figure and save _pair_layout_demo.png next to this file",
    )
    args = p.parse_args()
    if args.demo:
        _demo()
    else:
        _print_usage()
