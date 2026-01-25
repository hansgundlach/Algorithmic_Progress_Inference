"""
GPQA-D Pareto Frontier Analysis (Epoch AI Style)

Publication-quality visualization showing:
- Frontier models (best at any price) with trend line
- Price-controlled trend lines ($10 and fitted price)
- Inline labels on lines
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def logit(p: np.ndarray | float) -> np.ndarray | float:
    """Convert probability to logit scale."""
    p_clipped = np.clip(p, 0.001, 0.999)
    return np.log(p_clipped / (1 - p_clipped))


def load_data(path: str) -> pd.DataFrame:
    """Load and prepare the data."""
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
    df["log_Price"] = np.log10(df["Price"])
    return df.sort_values("Release Date").reset_index(drop=True)


def get_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pareto frontier models at each point in time.

    This function tracks the historical evolution of the Pareto frontier:
    - A model is included if it was on the Pareto frontier at the time it was released
    - Once included, it remains included even if later models dominate it
    - This shows how the "best available at each price point" evolved over time

    Returns all models that were ever on the Pareto frontier at their release date.
    """
    df_work = df.copy().sort_values("Release Date")
    pareto_indices = []

    for date in df_work["Release Date"].unique():
        # Get all models available up to this date
        available_models = df_work[df_work["Release Date"] <= date].copy()
        available_models = available_models.sort_values(["Price", "Score"])

        # Calculate the Pareto frontier from all available models up to this date
        frontier_indices = []

        for i, row in available_models.iterrows():
            # Check if this model is dominated by any model already on the frontier
            dominated = False
            for j in frontier_indices:
                frontier_row = available_models.loc[j]
                # Model is dominated if another model has:
                # - Better or equal score AND
                # - Lower or equal price AND
                # - At least one is strictly better
                if (
                    frontier_row["Score"] >= row["Score"]
                    and frontier_row["Price"] <= row["Price"]
                    and (
                        frontier_row["Score"] > row["Score"]
                        or frontier_row["Price"] < row["Price"]
                    )
                ):
                    dominated = True
                    break

            if not dominated:
                # Add this model to frontier
                frontier_indices.append(i)
                # Remove any previously added models that are now dominated by this one
                new_frontier_indices = []
                for j in frontier_indices[:-1]:
                    frontier_row = available_models.loc[j]
                    if not (
                        row["Score"] >= frontier_row["Score"]
                        and row["Price"] <= frontier_row["Price"]
                        and (
                            row["Score"] > frontier_row["Score"]
                            or row["Price"] < frontier_row["Price"]
                        )
                    ):
                        new_frontier_indices.append(j)
                frontier_indices = new_frontier_indices + [i]

        # Add models from CURRENT date that are on the frontier at this moment
        # This ensures we capture models that were frontier models when released,
        # even if they get dominated later
        current_date_models = df_work[df_work["Release Date"] == date]
        for i, row in current_date_models.iterrows():
            if i in frontier_indices:
                pareto_indices.append(i)

    # Remove duplicates and return
    pareto_indices = list(set(pareto_indices))
    return (
        df_work.loc[pareto_indices]
        .copy()
        .sort_values("Release Date")
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/price_reduction_models.csv")
    parser.add_argument("--out", default="../figures/gpqa_pareto_frontier.png")
    args = parser.parse_args()

    df = load_data(args.data)

    # Get Pareto frontier
    df_pareto = get_pareto_frontier(df)

    # Calculate time since start
    min_date = df["Release Date"].min()
    df_pareto["Years_Since_Start"] = (
        df_pareto["Release Date"] - min_date
    ).dt.days / 365.25

    # =========================================================================
    # STYLE PARAMETERS
    # =========================================================================
    FONT_SIZE_BASE = 9
    FONT_SIZE_LABEL = 9
    FONT_SIZE_TICK = 7.5
    FONT_SIZE_LEGEND = 7.5
    FONT_SIZE_ANNOTATION = 10
    LABEL_FONTSIZE = 10

    EXTRAPOLATION_MONTHS = 4
    PRICE_10 = 10.0

    FIG_WIDTH = 3.25
    FIG_HEIGHT = 2.8

    # Colors
    COLOR_FRONTIER = "#440154"  # Deep purple for frontier
    COLOR_PRICE_LINES = "#21918c"  # Teal for price lines
    COLOR_PARETO = "#35b779"  # Bright green for pareto points

    # =========================================================================
    # SLOPE ANNOTATION POSITIONS (easily adjustable)
    # =========================================================================
    # Frontier slope label: (x_offset, y_offset) in points from arrow target
    FRONTIER_SLOPE_OFFSET = (-80, -10)
    # Price-controlled slope label: (x_offset, y_offset) in points from arrow target
    PRICE_SLOPE_OFFSET = (50, -25)
    # Position along trend line (0.0 = start, 1.0 = end)
    SLOPE_ANNO_POSITION = 0.65

    # =========================================================================
    # Apply style settings
    # =========================================================================
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": FONT_SIZE_BASE,
            "axes.labelsize": FONT_SIZE_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelpad": 4,
        }
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=150)

    # =========================================================================
    # COMPUTE MODELS
    # =========================================================================

    # Prepare regression data for price-controlled model
    X_time = df_pareto["Years_Since_Start"].values.reshape(-1, 1)
    X_price = df_pareto["log_Price"].values.reshape(-1, 1)
    X_time_price = df_pareto[["Years_Since_Start", "log_Price"]].values
    y_logit = df_pareto["Score_logit"].values

    # Model: score ~ time + log(price) for price-controlled trends
    model_price_controlled = LinearRegression().fit(X_time_price, y_logit)

    # Calculate price-residualized points for pareto scatter
    # Step 1: Regress score on price only to get price effect
    model_price_only = LinearRegression().fit(X_price, y_logit)
    price_effect = model_price_only.predict(X_price)

    # Step 2: Remove price effect from scores (residualize), centered at mean
    y_residualized = y_logit - price_effect + np.mean(y_logit)

    # Calculate the "genuine fit" price - median price of pareto models
    median_price = df_pareto["Price"].median()
    log_median_price = np.log10(median_price)

    # Calculate frontier (best model over time, NOT price adjusted)
    df_sorted = df.sort_values("Release Date").copy()
    frontier_dates = []
    frontier_scores = []
    current_best = -np.inf

    for date in df_sorted["Release Date"].unique():
        date_models = df_sorted[df_sorted["Release Date"] == date]
        best_score_today = date_models["Score_logit"].max()

        if best_score_today > current_best:
            frontier_dates.append(date)
            frontier_scores.append(best_score_today)
            current_best = best_score_today

    frontier_dates = pd.to_datetime(frontier_dates)
    frontier_scores = np.array(frontier_scores)
    frontier_years = (frontier_dates - min_date).days / 365.25

    # Fit frontier trend (time only, no price control)
    X_frontier = frontier_years.values.reshape(-1, 1)
    model_frontier = LinearRegression().fit(X_frontier, frontier_scores)

    # =========================================================================
    # DATE RANGE FOR TREND LINES
    # =========================================================================

    max_date = df_pareto["Release Date"].max()
    max_date_extended = max_date + pd.DateOffset(months=EXTRAPOLATION_MONTHS)
    dates_range = pd.date_range(min_date, max_date_extended, periods=100)
    years_range = (dates_range - min_date).days / 365.25
    extrapolation_start_idx = np.argmin(np.abs(dates_range - max_date))

    # =========================================================================
    # COMPUTE TREND LINE PREDICTIONS
    # =========================================================================

    # Frontier trend (purple)
    pred_frontier_logit = model_frontier.predict(years_range.values.reshape(-1, 1))

    # Price-controlled predictions (teal)
    log_price_10 = np.log10(PRICE_10)
    pred_price_10_logit = (
        model_price_controlled.intercept_
        + model_price_controlled.coef_[0] * years_range.values
        + model_price_controlled.coef_[1] * log_price_10
    )

    pred_median_price_logit = (
        model_price_controlled.intercept_
        + model_price_controlled.coef_[0] * years_range.values
        + model_price_controlled.coef_[1] * log_median_price
    )

    # =========================================================================
    # PLOT DATA POINTS
    # =========================================================================

    # Plot frontier points (purple stars)
    ax.scatter(
        frontier_dates,
        frontier_scores,
        alpha=0.9,
        s=60,
        color=COLOR_FRONTIER,
        edgecolors="white",
        linewidth=0.5,
        label="Frontier models",
        zorder=5,
        marker="*",
    )

    # Plot pareto points (green circles) - price-residualized
    ax.scatter(
        df_pareto["Release Date"],
        y_residualized,
        alpha=0.5,
        s=25,
        color=COLOR_PARETO,
        edgecolors="white",
        linewidth=0.3,
        label="Price-adjusted (pareto)",
        zorder=4,
    )

    # =========================================================================
    # PLOT TREND LINES
    # =========================================================================

    # Frontier trend line (purple) - solid then dashed
    ax.plot(
        dates_range[: extrapolation_start_idx + 1],
        pred_frontier_logit[: extrapolation_start_idx + 1],
        "-",
        color=COLOR_FRONTIER,
        linewidth=2,
        alpha=0.9,
        zorder=3,
    )
    ax.plot(
        dates_range[extrapolation_start_idx:],
        pred_frontier_logit[extrapolation_start_idx:],
        "--",
        color=COLOR_FRONTIER,
        linewidth=2,
        alpha=0.6,
        zorder=3,
    )

    # $10 trend line (teal) - solid then dashed
    ax.plot(
        dates_range[: extrapolation_start_idx + 1],
        pred_price_10_logit[: extrapolation_start_idx + 1],
        "-",
        color=COLOR_PRICE_LINES,
        linewidth=2,
        alpha=0.9,
        zorder=3,
    )
    ax.plot(
        dates_range[extrapolation_start_idx:],
        pred_price_10_logit[extrapolation_start_idx:],
        "--",
        color=COLOR_PRICE_LINES,
        linewidth=2,
        alpha=0.6,
        zorder=3,
    )

    # Median price trend line (teal) - solid then dashed
    ax.plot(
        dates_range[: extrapolation_start_idx + 1],
        pred_median_price_logit[: extrapolation_start_idx + 1],
        "-",
        color=COLOR_PRICE_LINES,
        linewidth=2,
        alpha=0.9,
        zorder=3,
    )
    ax.plot(
        dates_range[extrapolation_start_idx:],
        pred_median_price_logit[extrapolation_start_idx:],
        "--",
        color=COLOR_PRICE_LINES,
        linewidth=2,
        alpha=0.6,
        zorder=3,
    )

    # Vertical line at extrapolation start
    ax.axvline(
        x=max_date, color="gray", linestyle=":", linewidth=1.2, alpha=0.4, zorder=0
    )

    # =========================================================================
    # INLINE PRICE LABELS (at end of lines, no arrows)
    # =========================================================================

    # Extend x-axis for labels
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.18)

    # Get end positions for labels
    end_idx = len(dates_range) - 1
    end_date = dates_range[end_idx]

    # $10 line label
    end_score_10 = pred_price_10_logit[end_idx]
    ax.annotate(
        "$10",
        xy=(end_date, end_score_10),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
        color=COLOR_PRICE_LINES,
        va="center",
        ha="left",
    )

    # Median price line label
    end_score_median = pred_median_price_logit[end_idx]
    price_label = f"${median_price:.2f}" if median_price < 1 else f"${median_price:.0f}"
    ax.annotate(
        price_label,
        xy=(end_date, end_score_median),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
        color=COLOR_PRICE_LINES,
        va="center",
        ha="left",
    )

    # =========================================================================
    # SLOPE ANNOTATIONS WITH ARROWS POINTING TO LINES
    # =========================================================================

    # Position for annotations along trend line
    anno_idx = int(len(dates_range) * SLOPE_ANNO_POSITION)
    anno_date = dates_range[anno_idx]

    # Frontier slope annotation (pointing to purple line)
    slope_frontier = model_frontier.coef_[0]
    anno_score_frontier = pred_frontier_logit[anno_idx]
    ax.annotate(
        f"{slope_frontier:.2f} logits/yr",
        xy=(anno_date, anno_score_frontier),
        xytext=FRONTIER_SLOPE_OFFSET,
        textcoords="offset points",
        fontsize=FONT_SIZE_ANNOTATION,
        color=COLOR_FRONTIER,
        fontweight="bold",
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=COLOR_FRONTIER, lw=1.0),
    )

    # Price-controlled slope annotation (pointing to teal line)
    slope_price = model_price_controlled.coef_[0]
    anno_score_price = pred_price_10_logit[anno_idx]
    ax.annotate(
        f"{slope_price:.2f} logits/yr",
        xy=(anno_date, anno_score_price),
        xytext=PRICE_SLOPE_OFFSET,
        textcoords="offset points",
        fontsize=FONT_SIZE_ANNOTATION,
        color=COLOR_PRICE_LINES,
        fontweight="bold",
        ha="right",
        va="top",
        arrowprops=dict(arrowstyle="->", color=COLOR_PRICE_LINES, lw=1.0),
    )

    # =========================================================================
    # FORMATTING
    # =========================================================================

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(
        ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=FONT_SIZE_TICK
    )
    ax.tick_params(axis="x", which="major", pad=2)

    ax.set_xlabel("Release Date", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("GPQA-D Score (logits)", fontsize=FONT_SIZE_LABEL)

    ax.grid(True, alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend with only two items
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=0.95,
        borderpad=0.4,
        handletextpad=0.4,
        fontsize=FONT_SIZE_LEGEND,
    )

    plt.tight_layout()

    # =========================================================================
    # SAVE
    # =========================================================================

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        args.out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"Saved: {args.out}")

    pdf_out = args.out.replace(".png", ".pdf")
    plt.savefig(pdf_out, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved: {pdf_out}")

    plt.show()

    # Print summary
    print(f"\nGPQA-D Pareto Analysis")
    print(f"Pareto models: {len(df_pareto)}")
    print(f"Frontier models: {len(frontier_dates)}")
    print(f"Median price of pareto models: ${median_price:.2f}")
    print(f"Frontier slope: {slope_frontier:.3f} logits/yr")
    print(f"Price-controlled slope: {slope_price:.3f} logits/yr")


if __name__ == "__main__":
    main()
