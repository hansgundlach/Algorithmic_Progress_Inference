"""
Best-Under-Budget Step Plots (3-panel, logit y-axis)

Creates a single figure with 3 subplots (GPQA-D, SWE-Bench, AIME), each showing:
- Unconstrained best-achievable performance over time ("No budget")
- Best-achievable performance under fixed budget thresholds (step functions)

Y-axis: logit(score), where score is a percentage in [0, 100].
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
    p_clipped = np.clip(p, 0.001, 0.999)
    return np.log(p_clipped / (1 - p_clipped))


def parse_budgets(s: str) -> list[float]:
    vals: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def load_benchmark(file_path: str, score_col: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["Release Date"] = pd.to_datetime(df["Release Date"])
    df["Score"] = df[score_col].astype(str).str.replace("%", "").astype(float)

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
        "--out",
        default="figures/best_under_budget_step_logit_3panel.png",
        help="Output figure path (default: figures/best_under_budget_step_logit_3panel.png)",
    )
    p.add_argument(
        "--gpqa-data",
        default="data/price_reduction_models.csv",
        help="GPQA-D CSV path (default: data/price_reduction_models.csv)",
    )
    p.add_argument(
        "--swe-data",
        default="data/swe_price_reduction_models.csv",
        help="SWE CSV path (default: data/swe_price_reduction_models.csv)",
    )
    p.add_argument(
        "--aime-data",
        default="data/aime_price_reduction_models.csv",
        help="AIME CSV path (default: data/aime_price_reduction_models.csv)",
    )
    p.add_argument(
        "--gpqa-budgets",
        default="0.1,1,10,50",
        help="Comma-separated GPQA budgets in USD (default: 0.1,1,10,50)",
    )
    p.add_argument(
        "--swe-budgets",
        default="50,200,500,1500",
        help="Comma-separated SWE budgets in USD (default: 50,200,500,1500)",
    )
    p.add_argument(
        "--aime-budgets",
        default="0.1,0.5,1,5",
        help="Comma-separated AIME budgets in USD (default: 0.1,0.5,1,5)",
    )
    return p.parse_args()


def plot_panel(ax, df: pd.DataFrame, title: str, budgets: list[float]) -> None:
    # Unconstrained
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

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Release Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("logit(score)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gpqa = load_benchmark(args.gpqa_data, "epoch_gpqa")
    swe = load_benchmark(args.swe_data, "epoch_swe")
    aime = load_benchmark(args.aime_data, "oneshot_AIME")

    gpqa_b = parse_budgets(args.gpqa_budgets)
    swe_b = parse_budgets(args.swe_budgets)
    aime_b = parse_budgets(args.aime_budgets)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    plot_panel(axes[0], gpqa, "GPQA-D: best under budget (logit)", gpqa_b)
    plot_panel(axes[1], swe, "SWE (epoch_swe): best under budget (logit)", swe_b)
    plot_panel(axes[2], aime, "AIME: best under budget (logit)", aime_b)

    # One shared legend (dedupe labels)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    fig.legend(uniq_h, uniq_l, loc="lower center", ncol=6, fontsize=9, title="Budgets")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
