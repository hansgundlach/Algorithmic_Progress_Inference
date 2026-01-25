"""
SWE-Bench Budget Step Plot (Epoch AI Style)

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
import matplotlib.dates as mdates
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
    df["Score"] = df["epoch_swe"].astype(str).str.replace("%", "").astype(float)
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
    parser.add_argument("--data", default="../data/swe_price_reduction_models.csv")
    parser.add_argument("--budgets", default="1, 10, 100, 1000")
    parser.add_argument("--out", default="../figures/swe_budget_epoch.png")
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

    # Also compute unconstrained
    unconstrained = best_under_budget(df, None)
    color_unc = "#9D7ED9"  # purple for "no budget"

    # Jitter offset for overlapping lines (in data units)
    # Calculate based on y-axis range
    all_scores = []
    for B in budgets:
        if len(budget_data[B]) > 0:
            all_scores.extend(budget_data[B]["Score_logit"].tolist())
    if len(unconstrained) > 0:
        all_scores.extend(unconstrained["Score_logit"].tolist())
    y_range = max(all_scores) - min(all_scores) if all_scores else 1
    JITTER_OFFSET = y_range * 0.015  # 1.5% of y-range

    # Helper function to get score at date using step interpolation
    def get_score_at_date(data_df, d):
        """Get score at a given date using step interpolation."""
        before = data_df[data_df["Date"] <= d]
        if len(before) == 0:
            return None
        return before.iloc[-1]["Score_logit"]

    # Plot lines from lowest budget to highest, then unconstrained
    # Higher budget lines get jitter offset where they overlap with lower lines
    all_lines_data = []  # Accumulate lines as we go

    for i, B in enumerate(budgets):
        sub = budget_data[B]
        if len(sub) == 0:
            continue

        trans = get_transitions(sub)
        color = EPOCH_COLORS[i % len(EPOCH_COLORS)]

        # Calculate jitter: offset this line where it overlaps with any line below
        jitter_amount = JITTER_OFFSET * (i + 1)  # Stack jitter for multiple overlaps

        dates = list(sub["Date"])
        scores = list(sub["Score_logit"])
        jittered_scores = []

        for j, (d, s) in enumerate(zip(dates, scores)):
            # Check if this point overlaps with any line below
            needs_jitter = False
            for below_df in all_lines_data:
                below_score = get_score_at_date(below_df, d)
                if below_score is not None and abs(s - below_score) < 1e-6:
                    needs_jitter = True
                    break

            if needs_jitter:
                jittered_scores.append(s + jitter_amount)
            else:
                jittered_scores.append(s)

        # Draw step line with jittered scores
        ax.step(
            dates,
            jittered_scores,
            where="post",
            linewidth=3.5,
            color=color,
            zorder=5 + i,
            solid_capstyle="butt",
        )

        # Add scatter markers at transition points (with jitter)
        trans_jittered = []
        for _, row in trans.iterrows():
            d = row["Date"]
            s = row["Score_logit"]
            # Find jittered score for this date
            idx = dates.index(d) if d in dates else -1
            if idx >= 0:
                trans_jittered.append(jittered_scores[idx])
            else:
                trans_jittered.append(s)

        ax.scatter(
            trans["Date"],
            trans_jittered,
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
                "end_score": jittered_scores[-1] if jittered_scores else sub["Score_logit"].iloc[-1],
            }
        )

        # Add this line's data to the list for subsequent overlap checks
        all_lines_data.append(sub)

    # Plot unconstrained frontier with jitter where it overlaps
    jitter_amount_unc = JITTER_OFFSET * (len(budgets) + 1)

    dates_unc = list(unconstrained["Date"])
    scores_unc = list(unconstrained["Score_logit"])
    jittered_scores_unc = []

    for d, s in zip(dates_unc, scores_unc):
        needs_jitter = False
        for below_df in all_lines_data:
            below_score = get_score_at_date(below_df, d)
            if below_score is not None and abs(s - below_score) < 1e-6:
                needs_jitter = True
                break

        if needs_jitter:
            jittered_scores_unc.append(s + jitter_amount_unc)
        else:
            jittered_scores_unc.append(s)

    ax.step(
        dates_unc,
        jittered_scores_unc,
        where="post",
        linewidth=3.5,
        color=color_unc,
        zorder=10,
        solid_capstyle="butt",
    )

    trans_unc = get_transitions(unconstrained)
    trans_jittered_unc = []
    for _, row in trans_unc.iterrows():
        d = row["Date"]
        s = row["Score_logit"]
        idx = dates_unc.index(d) if d in dates_unc else -1
        if idx >= 0:
            trans_jittered_unc.append(jittered_scores_unc[idx])
        else:
            trans_jittered_unc.append(s)

    ax.scatter(
        trans_unc["Date"],
        trans_jittered_unc,
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
            "end_score": jittered_scores_unc[-1] if jittered_scores_unc else unconstrained["Score_logit"].iloc[-1],
        }
    )

    # =========================================================================
    # STYLING
    # =========================================================================

    ax.set_xlabel("Date", fontsize=18, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel(
        "SWE-Bench Verified (logit)", fontsize=18, fontweight="bold", color=TEXT_COLOR
    )
    ax.set_title(
        "Best Achievable Performance Under Budget Constraints",
        fontsize=20,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=15,
    )

    # X-axis formatting (matching frontier_price.ipynb style)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.tick_params(axis="x", which="major", pad=2)

    # Extend x-axis for inline labels
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.15)

    # =========================================================================
    # INLINE LABELS (Epoch AI style - labels at end of lines)
    # =========================================================================

    # Sort line_data by end_score to avoid overlap, then add labels
    line_data_sorted = sorted(line_data, key=lambda x: x["end_score"], reverse=True)

    # Calculate vertical offsets to prevent label overlap
    # Sort by score and assign staggered offsets for close labels
    min_gap = 0.15  # minimum gap in logit units
    label_offsets = {}
    prev_score = None
    offset_direction = -1
    for item in line_data_sorted:
        if prev_score is not None and abs(item["end_score"] - prev_score) < min_gap:
            label_offsets[item["label"]] = offset_direction * 15
            offset_direction *= -1  # alternate direction
        else:
            label_offsets[item["label"]] = 0
        prev_score = item["end_score"]

    for item in line_data_sorted:
        vertical_offset = label_offsets.get(item["label"], 0)
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
