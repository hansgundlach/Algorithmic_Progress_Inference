"""
GPQA-D Budget Step Plot (Epoch AI Style)

Publication-quality visualization styled after epochai.org plots:
- Distinct categorical colors (teal, orange, pink, blue, purple)
- Inline labels on lines (no legend box)
- Clean minimal aesthetic with white background
- Subtle gridlines
- Markers at transition points
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator


# Epoch AI categorical color palette
EPOCH_COLORS = [
    "#19B5B5",  # teal/cyan
    "#F28E2B",  # orange
    "#E15989",  # pink/magenta
    "#4E79A7",  # steel blue
    "#9D7ED9",  # purple
    "#59A14F",  # green
]


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p_clipped = np.clip(p, 0.001, 0.999)
    return np.log(p_clipped / (1 - p_clipped))


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Release Date"] = pd.to_datetime(df["Release Date"])
    df["Score"] = df["epoch_gpqa"].astype(str).str.replace("%", "").astype(float)
    df["Price"] = (
        df["Benchmark Cost USD"].astype(str).str.replace("[$,]", "", regex=True)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df[["Model", "Release Date", "Score", "Price"]].dropna()
    df = df[(df["Price"] > 0) & (df["Score"] > 0)].copy()
    df["Score_logit"] = logit(df["Score"] / 100)
    return df.sort_values("Release Date").reset_index(drop=True)


def best_under_budget(df: pd.DataFrame, budget: float | None) -> pd.DataFrame:
    dates = sorted(df["Release Date"].unique())
    rows = []
    for d in dates:
        avail = df[df["Release Date"] <= d]
        if budget is not None:
            avail = avail[avail["Price"] <= budget]
        if len(avail) == 0:
            continue
        best = avail.loc[avail["Score_logit"].idxmax()]
        rows.append(
            {
                "Date": d,
                "Score_logit": float(best["Score_logit"]),
                "Score": float(best["Score"]),
                "Model": best["Model"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/price_reduction_models.csv")
    parser.add_argument("--budgets", default="0.1,1,10")
    parser.add_argument("--out", default="figures/gpqa_budget_epoch.png")
    args = parser.parse_args()

    budgets = [float(b.strip()) for b in args.budgets.split(",")]
    df = load_data(args.data)

    # =========================================================================
    # EPOCH AI STYLE CONFIGURATION
    # =========================================================================

    BACKGROUND = "#FFFFFF"
    GRID_COLOR = "#E8E8E8"
    TEXT_COLOR = "#333333"
    AXIS_COLOR = "#999999"
    LABEL_FONTSIZE = 20  # Font size for budget labels ($0.1, $1, $10, etc.)

    # Reset and configure style
    plt.rcdefaults()
    plt.rcParams.update(
        {
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "sans-serif",
            ],
            "font.size": 11,
            "font.weight": "regular",
            # Axes
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": GRID_COLOR,
            "axes.linewidth": 0.5,
            "axes.labelcolor": TEXT_COLOR,
            "axes.labelsize": 12,
            "axes.labelweight": "regular",
            "axes.titlesize": 14,
            "axes.titleweight": "medium",
            "axes.titlepad": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            # Grid - both axes, subtle
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            # Ticks - larger and darker for ICML readability
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            # Figure
            "figure.facecolor": BACKGROUND,
            "figure.dpi": 150,
        }
    )

    # =========================================================================
    # CREATE FIGURE
    # =========================================================================

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Find transition points for markers
    def get_transitions(data):
        if len(data) <= 1:
            return data
        transitions = [data.iloc[0]]
        for i in range(1, len(data)):
            if data.iloc[i]["Score_logit"] != data.iloc[i - 1]["Score_logit"]:
                transitions.append(data.iloc[i])
        return pd.DataFrame(transitions)

    # Store line data for inline labels
    line_data = []

    # First, compute all budget-constrained data
    budget_data = {}
    for B in budgets:
        budget_data[B] = best_under_budget(df, B)

    # Plot each budget constraint with categorical colors (solid lines)
    for i, B in enumerate(budgets):
        sub = budget_data[B]
        if len(sub) == 0:
            continue

        trans = get_transitions(sub)
        color = EPOCH_COLORS[i % len(EPOCH_COLORS)]

        ax.step(
            sub["Date"],
            sub["Score_logit"],
            where="post",
            linewidth=3.5,
            color=color,
            zorder=5 + i,
        )
        ax.scatter(
            trans["Date"],
            trans["Score_logit"],
            s=70,
            color=color,
            zorder=6 + i,
            edgecolors="white",
            linewidths=2,
        )

        label = f"${B:g}" if B >= 1 else f"${B:.2f}"
        line_data.append(
            {
                "label": label,
                "color": color,
                "end_date": sub["Date"].iloc[-1],
                "end_score": sub["Score_logit"].iloc[-1],
            }
        )

    # Plot unconstrained frontier (no budget) - use Epoch blue
    # Draw dashed where it overlaps with highest budget, solid otherwise
    unconstrained = best_under_budget(df, None)
    trans_unc = get_transitions(unconstrained)
    color_unc = "#4E79A7"  # steel blue for "no budget"

    # Get highest budget data for overlap detection
    highest_budget = max(budgets)
    highest_budget_data = budget_data[highest_budget]

    # Create a lookup for highest budget scores by date
    hb_scores = dict(
        zip(highest_budget_data["Date"], highest_budget_data["Score_logit"])
    )

    # Draw segments: dashed where overlapping, solid otherwise
    dates = list(unconstrained["Date"])
    scores = list(unconstrained["Score_logit"])

    for j in range(len(dates) - 1):
        d1, d2 = dates[j], dates[j + 1]
        s1 = scores[j]

        # Check if this segment overlaps with highest budget line
        hb_score = hb_scores.get(d1, None)
        is_overlap = hb_score is not None and abs(s1 - hb_score) < 0.01

        linestyle = (0, (4, 2)) if is_overlap else "-"

        ax.plot(
            [d1, d2],
            [s1, s1],
            linewidth=3.5,
            linestyle=linestyle,
            color=color_unc,
            zorder=10,
            solid_capstyle="butt",
        )
        # Vertical segment for step
        if j < len(dates) - 2:
            s2 = scores[j + 1]
            ax.plot(
                [d2, d2],
                [s1, s2],
                linewidth=3.5,
                linestyle=linestyle,
                color=color_unc,
                zorder=10,
            )

    ax.scatter(
        trans_unc["Date"],
        trans_unc["Score_logit"],
        s=70,
        color=color_unc,
        zorder=11,
        edgecolors="white",
        linewidths=2,
    )
    line_data.append(
        {
            "label": "Any Budget",
            "color": color_unc,
            "end_date": unconstrained["Date"].iloc[-1],
            "end_score": unconstrained["Score_logit"].iloc[-1],
        }
    )

    # =========================================================================
    # STYLING
    # =========================================================================

    ax.set_xlabel("Date", fontsize=18, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel(
        "GPQA-D Accuracy (logit)", fontsize=18, fontweight="bold", color=TEXT_COLOR
    )

    # X-axis formatting
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))

    # Extend x-axis for inline labels
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.15)

    # =========================================================================
    # INLINE LABELS (Epoch AI style - labels at end of lines)
    # =========================================================================

    # Sort line_data by end_score to avoid overlap, then add labels
    line_data_sorted = sorted(line_data, key=lambda x: x["end_score"], reverse=True)

    for item in line_data_sorted:
        # Adjust vertical offset for $10 label to avoid overlap
        vertical_offset = -8 if item["label"] == "$10" else 0
        ax.annotate(
            item["label"],
            xy=(item["end_date"], item["end_score"]),
            xytext=(12, vertical_offset),
            textcoords="offset points",
            fontsize=LABEL_FONTSIZE,
            fontweight="bold",
            color=item["color"],
            va="center",
            ha="left",
        )

    # =========================================================================
    # FINAL ADJUSTMENTS
    # =========================================================================

    # Style spines for visibility
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["left"].set_linewidth(1.5)

    plt.tight_layout()

    # Save PNG and PDF (for ICML)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        args.out,
        dpi=300,
        bbox_inches="tight",
        facecolor=BACKGROUND,
        edgecolor="none",
    )
    print(f"Saved: {args.out}")

    # Also save PDF version for ICML
    pdf_out = args.out.replace(".png", ".pdf")
    plt.savefig(
        pdf_out,
        bbox_inches="tight",
        facecolor=BACKGROUND,
        edgecolor="none",
    )
    print(f"Saved: {pdf_out}")

    plt.show()


if __name__ == "__main__":
    main()
