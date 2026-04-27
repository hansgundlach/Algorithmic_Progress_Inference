"""
Separate Pareto Frontiers: MoE vs Dense Models (AIME)

Plots MoE and Dense Pareto frontiers on the same axes (cost vs error rate).
MoE is derived from the "Known Active Parameters" column being non-null.

Run from budget_plots_epoch/ or pass --data / --out with explicit paths.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_sys_dir = Path(__file__).resolve().parent
if str(_sys_dir) not in sys.path:
    sys.path.insert(0, str(_sys_dir))

from aime_pareto_pair_layout import (  # noqa: E402
    PAIR_FIGSIZE_INCHES,
    PAIR_XMIN,
    PAIR_XMAX,
    PAIR_YMIN,
    PAIR_YMAX,
    apply_pair_layout,
    savefig_pair,
)

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import pandas as pd


def calculate_pareto_frontier(
    df: pd.DataFrame, price_col: str, error_col: str
) -> pd.DataFrame:
    if len(df) == 0:
        return df

    df_work = df.reset_index(drop=True).copy()
    pareto_indices = []

    for i in range(len(df_work)):
        is_dominated = False
        row_i = df_work.iloc[i]

        for j in range(len(df_work)):
            if i == j:
                continue
            row_j = df_work.iloc[j]
            if (
                row_j[price_col] <= row_i[price_col]
                and row_j[error_col] <= row_i[error_col]
                and (
                    row_j[price_col] < row_i[price_col]
                    or row_j[error_col] < row_i[error_col]
                )
            ):
                is_dominated = True
                break

        if not is_dominated:
            pareto_indices.append(i)

    return df_work.loc[pareto_indices].sort_values(price_col).copy()


def clean_model_name(name) -> str:
    if pd.isna(name):
        return ""
    name = str(name)
    name = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", "", name)
    name = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", "", name)
    name = re.sub(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}\b",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(
        r"\b\d{4}\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"\s+20\d{2}\s*$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"[-_/]\s*$", "", name).strip()
    return name


def plot_separate_moe_dense_frontiers(
    df,
    price_col="Benchmark Cost USD",
    benchmark_col="oneshot_AIME",
    moe_col="MoE",
    show_model_names=False,
    open_license_only=False,
    extend_frontier=True,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    figsize=PAIR_FIGSIZE_INCHES,
    title=None,
    save_path=None,
    show_plot=True,
):
    TICK_LABELSIZE = 12
    AXIS_LABELSIZE = 14
    AXIS_LABELWEIGHT = "bold"
    TITLE_FONTSIZE = 16
    TITLE_FONTWEIGHT = "bold"
    TITLE_PAD = 8
    LEGEND_FONTSIZE = 11
    ANNOTATE_FONTSIZE = 9
    POINT_SIZE_SCATTER = 80
    PARETO_POINT_SIZE = 100

    df_work = df.copy()

    df_work[benchmark_col] = (
        df_work[benchmark_col].astype(str).str.replace("%", "", regex=False).str.strip()
    )
    df_work[benchmark_col] = pd.to_numeric(df_work[benchmark_col], errors="coerce")
    df_work["error_rate"] = 1 - (df_work[benchmark_col] / 100)

    df_work[price_col] = (
        df_work[price_col].astype(str).str.replace("[$,]", "", regex=True)
    )
    df_work[price_col] = pd.to_numeric(df_work[price_col], errors="coerce")

    df_work["Release Date"] = pd.to_datetime(df_work["Release Date"])

    if open_license_only:
        df_work = df_work[
            df_work["License"].notna()
            & df_work["License"].str.contains("open", case=False, na=False)
        ]

    df_work = df_work.dropna(
        subset=["Release Date", price_col, "error_rate", benchmark_col, moe_col]
    )
    df_work = df_work[df_work[price_col] > 0]
    df_work = df_work[df_work[benchmark_col] > 0]

    if len(df_work) == 0:
        print("No valid data to plot after filtering")
        return None, None

    max_price = xmax if xmax is not None else df_work[price_col].max()
    max_error = ymax if ymax is not None else df_work["error_rate"].max()

    df_moe = df_work[df_work[moe_col] == True].copy()
    df_dense = df_work[df_work[moe_col] == False].copy()

    print(f"Total models: {len(df_work)}")
    print(f"MoE models: {len(df_moe)}")
    print(f"Dense models: {len(df_dense)}")

    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
        }
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    moe_color = "#E74C3C"
    dense_color = "#3498DB"

    if len(df_moe) > 0:
        ax.scatter(
            df_moe[price_col],
            df_moe["error_rate"],
            c=moe_color,
            s=POINT_SIZE_SCATTER,
            alpha=0.6,
            edgecolors="white",
            linewidth=1.5,
            label="MoE Models",
            zorder=3,
        )

    if len(df_dense) > 0:
        ax.scatter(
            df_dense[price_col],
            df_dense["error_rate"],
            c=dense_color,
            s=POINT_SIZE_SCATTER,
            alpha=0.6,
            edgecolors="white",
            linewidth=1.5,
            label="Dense Models",
            zorder=3,
        )

    for df_group, color, label_prefix in [
        (df_moe, moe_color, "MoE"),
        (df_dense, dense_color, "Dense"),
    ]:
        if len(df_group) == 0:
            continue

        pareto = calculate_pareto_frontier(df_group, price_col, "error_rate")
        print(f"\n{label_prefix} Pareto frontier points: {len(pareto)}")

        if len(pareto) == 0:
            continue

        if extend_frontier:
            pareto_extended = pareto.copy()
            leftmost = pareto.iloc[0]
            if leftmost["error_rate"] < max_error:
                vertical_point = pd.DataFrame(
                    [
                        {
                            price_col: leftmost[price_col],
                            "error_rate": max_error,
                            "Model": "",
                        }
                    ]
                )
                pareto_extended = pd.concat(
                    [vertical_point, pareto_extended], ignore_index=True
                )

            rightmost = pareto.iloc[-1]
            if rightmost[price_col] < max_price:
                extend_x = max_price if xmax is not None else max_price * 1.5
                horizontal_point = pd.DataFrame(
                    [
                        {
                            price_col: extend_x,
                            "error_rate": rightmost["error_rate"],
                            "Model": "",
                        }
                    ]
                )
                pareto_extended = pd.concat(
                    [pareto_extended, horizontal_point], ignore_index=True
                )

            pareto_plot = pareto_extended
        else:
            pareto_plot = pareto

        prices = pareto_plot[price_col].values
        errors = pareto_plot["error_rate"].values
        step_x, step_y = [], []
        for i in range(len(prices)):
            if i == 0:
                step_x.append(prices[i])
                step_y.append(errors[i])
            else:
                step_x.append(prices[i])
                step_y.append(errors[i - 1])
                step_x.append(prices[i])
                step_y.append(errors[i])

        ax.plot(
            step_x,
            step_y,
            color=color,
            linestyle="-",
            linewidth=3,
            label=f"{label_prefix} Pareto Frontier",
            zorder=5,
            alpha=0.9,
        )

        ax.scatter(
            pareto[price_col],
            pareto["error_rate"],
            c=color,
            s=PARETO_POINT_SIZE,
            marker="o",
            edgecolors="white",
            linewidth=2.5,
            zorder=6,
            alpha=1.0,
        )

        if show_model_names:
            labels = []
            for _, row in pareto.iterrows():
                cn = clean_model_name(row["Model"])
                if cn:
                    labels.append(
                        {"name": cn, "x": row[price_col], "y": row["error_rate"]}
                    )

            labels.sort(key=lambda d: d["y"])
            placed_labels = []

            for label_info in labels:
                x_data, y_data, name = (
                    label_info["x"],
                    label_info["y"],
                    label_info["name"],
                )
                x_offset, y_offset = -10, -5

                for placed in placed_labels:
                    if abs(y_data - placed["y"]) < 0.025:
                        y_offset = -23 if y_offset > 0 else 10

                placed_labels.append({"x": x_data, "y": y_data})
                ax.annotate(
                    name,
                    (x_data, y_data),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    fontsize=ANNOTATE_FONTSIZE,
                    fontweight="medium",
                    alpha=0.9,
                    ha="right",
                    va="center",
                    color=color,
                    zorder=7,
                )

    ax.set_xscale("log")

    if xmin is not None or xmax is not None:
        current_xlim = ax.get_xlim()
        ax.set_xlim(
            left=xmin if xmin is not None else current_xlim[0],
            right=xmax if xmax is not None else current_xlim[1],
        )

    if ymin is not None or ymax is not None:
        current_ylim = ax.get_ylim()
        ax.set_ylim(
            bottom=ymin if ymin is not None else current_ylim[0],
            top=ymax if ymax is not None else current_ylim[1],
        )

    def dollar_formatter(x, pos):
        if x >= 1:
            return f"${x:.0f}"
        elif x >= 0.01:
            return f"${x:.2f}"
        elif x >= 0.001:
            return f"${x:.3f}"
        else:
            return f"${x:.4f}"

    ax.xaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_xlabel(
        "Benchmark Cost (USD)", fontsize=AXIS_LABELSIZE, fontweight=AXIS_LABELWEIGHT
    )
    ax.set_ylabel("Error Rate", fontsize=AXIS_LABELSIZE, fontweight=AXIS_LABELWEIGHT)

    if title is None:
        title = "MoE vs Dense Pareto Frontiers\nAIME"
        if open_license_only:
            title += " (Open Weight Only)"

    ax.set_title(
        title, fontsize=TITLE_FONTSIZE, fontweight=TITLE_FONTWEIGHT, pad=TITLE_PAD
    )

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.legend(
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="gray",
    )

    ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("gray")
        spine.set_linewidth(0.5)

    # Fixed margins — must match aime_pareto_names.py for identical axes boxes
    apply_pair_layout(fig)

    if save_path is not None:
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        savefig_pair(fig, save_path, dpi=300)
        pdf_path = save_path.replace(".png", ".pdf")
        savefig_pair(fig, pdf_path, dpi=300)
        print(f"\nFigure saved to: {save_path}")
        print(f"Figure saved to: {pdf_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoE vs Dense separate Pareto frontiers (AIME)."
    )
    parser.add_argument("--data", default="../data/aime_price_reduction_models.csv")
    parser.add_argument("--out", default="../figures/moe_dense_pareto_aime.png")
    parser.add_argument("--show-model-names", action="store_true")
    parser.add_argument(
        "--no-open-only",
        action="store_true",
        help="Include all licenses (default: open only)",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    # Only include models with known parameter counts (required for MoE vs Dense classification)
    df = df[df["Parameters"].notna()].copy()
    df["MoE"] = df["Known Active Parameters"].notna()

    title = "MoE vs Dense Pareto Frontiers\nAIME (Open Weight Only)"
    if args.no_open_only:
        title = "MoE vs Dense Pareto Frontiers\nAIME"

    plot_separate_moe_dense_frontiers(
        df,
        price_col="Benchmark Cost USD",
        benchmark_col="oneshot_AIME",
        moe_col="MoE",
        show_model_names=args.show_model_names,
        open_license_only=not args.no_open_only,
        extend_frontier=True,
        xmin=PAIR_XMIN,
        xmax=PAIR_XMAX,
        ymin=PAIR_YMIN,
        ymax=PAIR_YMAX,
        title=title,
        save_path=str(Path(args.out).resolve()),
        show_plot=not args.no_show,
    )
    # plot_separate_moe_dense_frontiers(
    #     df,
    #     price_col="Benchmark Cost USD",
    #     benchmark_col="oneshot_AIME",
    #     moe_col="MoE",
    #     show_model_names=True,
    #     open_license_only=not args.no_open_only,
    #     extend_frontier=True,
    #     xmin=PAIR_XMIN,
    #     xmax=PAIR_XMAX,
    #     ymin=PAIR_YMIN,
    #     ymax=PAIR_YMAX,
    #     title=title,
    #     save_path=str(Path(args.out).resolve()),
    #     show_plot=not args.no_show,
    # )


if __name__ == "__main__":
    main()
