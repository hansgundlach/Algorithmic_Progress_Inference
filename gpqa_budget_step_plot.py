"""
GPQA-D Best-Under-Budget Step Plot (logit y-axis)

Generates a step-function plot of best achievable GPQA-D performance over time under
fixed budget constraints, plus an unconstrained "No budget" curve.

Y-axis: logit(Score), where Score is the benchmark percentage in [0, 100].
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def logit(p: np.ndarray | float) -> np.ndarray | float:
    """Convert probability to logit scale, with clipping for numerical stability."""
    p_clipped = np.clip(p, 0.001, 0.999)
    return np.log(p_clipped / (1 - p_clipped))


def load_gpqa(file_path: str) -> pd.DataFrame:
    """Load and clean GPQA-D data needed for the plot."""
    df = pd.read_csv(file_path)
    df["Release Date"] = pd.to_datetime(df["Release Date"])
    df["Score"] = df["epoch_gpqa"].astype(str).str.replace("%", "").astype(float)

    df["Price"] = (
        df["Benchmark Cost USD"].astype(str).str.replace("[$,]", "", regex=True)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df[["Model", "Release Date", "Score", "Price"]].dropna()
    df = df[(df["Price"] > 0) & (df["Score"] > 0)].copy()

    df["Score_logit"] = logit(df["Score"] / 100)

    min_date_ordinal = df["Release Date"].min().toordinal()
    df["Years_Since_Start"] = (
        df["Release Date"].map(datetime.toordinal) - min_date_ordinal
    ) / 365.25

    return df.sort_values("Release Date").reset_index(drop=True)


def best_over_time_step(df: pd.DataFrame, budget: float | None) -> pd.DataFrame:
    """
    Compute best-achievable step series over time.

    If budget is None => unconstrained (no budget).
    Else => only consider models with Price <= budget.
    """
    dates = sorted(df["Release Date"].unique())
    rows = []
    for d in dates:
        avail = df[df["Release Date"] <= d]
        if budget is not None:
            avail = avail[avail["Price"] <= budget]
        if len(avail) == 0:
            continue
        best_idx = avail["Score_logit"].idxmax()
        best = avail.loc[best_idx]
        rows.append(
            {
                "Release Date": d,
                "Best Score (logit)": float(best["Score_logit"]),
                "Best Score (%)": float(best["Score"]),
                "Best Model": best["Model"],
                "Best Model Price": float(best["Price"]),
            }
        )
    return pd.DataFrame(rows).sort_values("Release Date")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default="data/price_reduction_models.csv",
        help="Path to GPQA-D dataset CSV (default: data/price_reduction_models.csv)",
    )
    p.add_argument(
        "--budgets",
        default="0.1,1,10,50",
        help="Comma-separated budgets in USD (default: 0.1,1,10,50)",
    )
    p.add_argument(
        "--out",
        default="figures/gpqa_best_under_budget_step_logit.png",
        help="Output figure path (default: figures/gpqa_best_under_budget_step_logit.png)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]

    df = load_gpqa(args.data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Unconstrained (implicit budget can increase)
    unconstrained = best_over_time_step(df, budget=None)
    ax.step(
        unconstrained["Release Date"],
        unconstrained["Best Score (logit)"],
        where="post",
        linewidth=2.5,
        linestyle="--",
        color="black",
        label="No budget",
    )

    # Fixed budgets
    for B in budgets:
        sub = best_over_time_step(df, budget=B)
        if len(sub) == 0:
            continue
        ax.step(
            sub["Release Date"],
            sub["Best Score (logit)"],
            where="post",
            linewidth=2,
            label=f"≤ ${B:g}",
        )

    ax.set_title(
        "GPQA-D: Best achievable under fixed budget over time (logit scale)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Release Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("logit(GPQA-D score)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Budget", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

