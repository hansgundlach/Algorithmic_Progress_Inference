"""
Multi-Benchmark Price Comparison (Epoch AI Style)

Publication-quality visualization showing the benchmark cost of the
best-performing model over time across GPQA-D, SWE-Bench, and AIME.
Styled for ICML dual-column format.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression


def load_and_prepare(path, score_col):
    """Load CSV, parse dates/prices, drop rows missing score or price."""
    df = pd.read_csv(path)
    df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
    df["Benchmark Cost USD"] = pd.to_numeric(
        df["Benchmark Cost USD"].astype(str).str.replace("[$,]", "", regex=True),
        errors="coerce",
    )
    df[score_col] = pd.to_numeric(
        df[score_col].astype(str).str.replace("%", ""), errors="coerce"
    )
    df = df.dropna(subset=["Release Date", "Benchmark Cost USD", score_col])
    df = df[df["Benchmark Cost USD"] > 0]
    return df


def plot_multi_benchmark_price_comparison(
    df_gpqa, df_swe, df_aime, min_date=None, save_path=None, scatter_alpha=0.65
):
    """
    Plot the benchmark price of best-performing models over time across
    multiple benchmarks.  For each benchmark, shows the price of models
    with the best performance up to that point.
    """

    # ========================================================================
    # EPOCH AI STYLE CONFIGURATION
    # ========================================================================
    FONT_SIZE_BASE = 9
    FONT_SIZE_TITLE = 10
    FONT_SIZE_LABEL = 9
    FONT_SIZE_TICK = 7.5
    FONT_SIZE_LEGEND = 7.0

    FIG_WIDTH = 3.25
    FIG_HEIGHT = 2.8

    SCATTER_POINT_SIZE = 40
    SCATTER_EDGE_WIDTH = 0.5
    TRENDLINE_WIDTH = 1.5

    GRID_ALPHA = 0.25
    GRID_WIDTH = 0.5

    COLOR_GPQA = "#440154"
    COLOR_SWE = "#FF7043"
    COLOR_AIME = "#35b779"

    # ========================================================================
    # Style
    # ========================================================================
    sns.set_style("white")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": FONT_SIZE_BASE,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelpad": 4,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    benchmarks = [
        {
            "name": "GPQA-D",
            "df": df_gpqa,
            "score_col": "epoch_gpqa",
            "color": COLOR_GPQA,
            "marker": "o",
        },
        {
            "name": "SWE-Bench",
            "df": df_swe,
            "score_col": "epoch_swe",
            "color": COLOR_SWE,
            "marker": "s",
        },
        {
            "name": "AIME",
            "df": df_aime,
            "score_col": "oneshot_AIME",
            "color": COLOR_AIME,
            "marker": "^",
        },
    ]

    for bench in benchmarks:
        df_work = bench["df"].copy()

        if min_date is not None:
            if isinstance(min_date, str):
                min_date_dt = pd.to_datetime(min_date)
            else:
                min_date_dt = min_date
            df_work = df_work[df_work["Release Date"] >= min_date_dt]

        df_work = df_work.sort_values(
            ["Release Date", bench["score_col"]], ascending=[True, False]
        )
        df_work = df_work.groupby("Release Date").first().reset_index()
        df_work["Date_Ordinal"] = df_work["Release Date"].map(datetime.toordinal)

        # Find best-performing model at each point in time (cumulative max)
        df_work["Is_Best"] = False
        prev_max = -np.inf
        for idx in df_work.index:
            score = df_work.loc[idx, bench["score_col"]]
            if score > prev_max:
                df_work.loc[idx, "Is_Best"] = True
                prev_max = score

        best = df_work[df_work["Is_Best"]].copy()
        if len(best) == 0:
            print(f"No best models found for {bench['name']}")
            continue

        # Trend line
        annual_factor = None
        if len(best) >= 2:
            X = best["Date_Ordinal"].values.reshape(-1, 1)
            y_log = np.log10(best["Benchmark Cost USD"].values + 0.01)
            model = LinearRegression().fit(X, y_log)

            x_range = np.arange(X.min(), X.max() + 1)
            x_dates = [datetime.fromordinal(int(d)) for d in x_range]
            y_pred = 10 ** model.predict(x_range.reshape(-1, 1)) - 0.01

            annual_factor = (10 ** model.coef_[0]) ** 365

            ax.plot(
                x_dates,
                y_pred,
                color=bench["color"],
                linestyle="--",
                linewidth=TRENDLINE_WIDTH,
                alpha=0.7,
                zorder=4,
                clip_on=True,
            )

        label = bench["name"]
        if annual_factor is not None:
            label = f"{bench['name']} ({annual_factor:.2f}x/yr)"

        ax.scatter(
            best["Release Date"],
            best["Benchmark Cost USD"],
            color=bench["color"],
            s=SCATTER_POINT_SIZE,
            marker=bench["marker"],
            alpha=scatter_alpha,
            edgecolor="white",
            linewidth=SCATTER_EDGE_WIDTH,
            label=label,
            zorder=5,
            clip_on=True,
        )

        print(f"\n{bench['name']} Summary:")
        print(f"  Best models: {len(best)}")
        print(
            f"  Price range: ${best['Benchmark Cost USD'].min():.2f}"
            f" to ${best['Benchmark Cost USD'].max():.2f}"
        )
        if annual_factor is not None:
            print(f"  Annual price factor: {annual_factor:.2f}x")

    # Axes
    ax.set_xlabel("Release Date", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Benchmark Cost (USD)", fontsize=FONT_SIZE_LABEL)
    ax.set_yscale("log")

    def dollar_fmt(x, _):
        if x >= 1:
            return f"${x:.0f}"
        elif x >= 0.1:
            return f"${x:.1f}"
        elif x >= 0.01:
            return f"${x:.2f}"
        return f"${x:.3f}"

    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(
        ax.xaxis.get_majorticklabels(),
        rotation=45,
        ha="right",
        fontsize=FONT_SIZE_TICK,
    )
    ax.tick_params(axis="x", which="major", pad=2)

    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_WIDTH, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left",
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        fancybox=False,
        framealpha=0.85,
        edgecolor="black",
        facecolor="white",
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
        ncol=1,
    )

    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK, width=0.8, length=3)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            edgecolor="none",
        )
        pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
        plt.savefig(
            pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"\nSaved: {save_path}")
        print(f"Saved: {pdf_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-benchmark price comparison plot"
    )
    parser.add_argument(
        "--gpqa", default="../data/gpqa_price_reduction_models.csv", help="GPQA CSV"
    )
    parser.add_argument(
        "--swe", default="../data/swe_price_reduction_models.csv", help="SWE CSV"
    )
    parser.add_argument(
        "--aime", default="../data/aime_price_reduction_models.csv", help="AIME CSV"
    )
    parser.add_argument(
        "--out",
        default="../figures/multi_benchmark_price_comparison.png",
        help="Output path",
    )
    parser.add_argument(
        "--min-date", default="2024-04-01", help="Minimum date filter (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    df_gpqa = load_and_prepare(args.gpqa, "epoch_gpqa")
    df_swe = load_and_prepare(args.swe, "epoch_swe")
    df_aime = load_and_prepare(args.aime, "oneshot_AIME")

    min_date = datetime.strptime(args.min_date, "%Y-%m-%d") if args.min_date else None

    plot_multi_benchmark_price_comparison(
        df_gpqa, df_swe, df_aime, min_date=min_date, save_path=args.out
    )


if __name__ == "__main__":
    main()
