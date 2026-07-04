import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = "results_data/regression_comparison_table_raw.csv"
OUTPUT_PDF = "figures/algorithmic_progress_bars_neurips.pdf"
OUTPUT_PNG = "figures/algorithmic_progress_bars_neurips.png"
HARDWARE_FACTOR = 1 / 0.7  # hardware costs drop 30%/yr → gain factor = 1/0.7 ≈ 1.4286

BENCHMARK_MAP = {
    "GPQA": "GPQA-D",
    "AIME": "AIME",
    "SWE-Bench": "SWE-Bench",
}
BENCHMARK_ORDER = ["GPQA-D", "SWE-Bench", "AIME"]
XTICK_LABELS = ["GPQA-D", "SWE-\nBench", "AIME"]


def parse_ci(ci_text):
    match = re.fullmatch(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", str(ci_text))
    if not match:
        raise ValueError(f"Could not parse CI: {ci_text}")
    return float(match.group(1)), float(match.group(2))


def load_open_license_estimates(path=RESULTS_PATH):
    df = pd.read_csv(path)
    df["Benchmark"] = df["Benchmark"].ffill()

    open_license_estimates = {}
    for _, row in df.iterrows():
        benchmark = str(row["Benchmark"]).strip()
        restriction = str(row["Restriction"]).strip()
        if benchmark not in BENCHMARK_MAP:
            continue
        if restriction != "Pareto Restricted Open License":
            continue

        factor = row["Year Decrease Factor"]
        if factor == "N/A":
            continue
        open_license_estimates[BENCHMARK_MAP[benchmark]] = {
            "factor": float(factor),
            "ci": parse_ci(row["90% CI"]),
        }

    missing = [bench for bench in BENCHMARK_ORDER if bench not in open_license_estimates]
    if missing:
        raise ValueError(f"Missing open-license estimates for: {', '.join(missing)}")

    return open_license_estimates


def compute_algorithmic_progress(open_license_estimates):
    return {
        benchmark: {
            "factor": open_license_estimates[benchmark]["factor"] / HARDWARE_FACTOR,
            "ci": tuple(
                endpoint / HARDWARE_FACTOR
                for endpoint in open_license_estimates[benchmark]["ci"]
            ),
        }
        for benchmark in BENCHMARK_ORDER
    }


def plot_algorithmic_progress(algorithmic_progress):
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

    values = [algorithmic_progress[bench]["factor"] for bench in BENCHMARK_ORDER]
    ci_bounds = [algorithmic_progress[bench]["ci"] for bench in BENCHMARK_ORDER]
    lower_errors = [value - lower for value, (lower, _) in zip(values, ci_bounds)]
    upper_errors = [upper - value for value, (_, upper) in zip(values, ci_bounds)]
    x = range(len(BENCHMARK_ORDER))
    color = mpl.colormaps["viridis"](0.5)

    fig, ax = plt.subplots(figsize=(3.0, 2.05))
    bars = ax.bar(x, values, color=color, edgecolor="none", width=0.58, zorder=3)
    ax.errorbar(
        list(x),
        values,
        yerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor="#333333",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        zorder=4,
    )

    ax.set_yscale("log")
    ax.set_ylim(0.9, max(upper for _, upper in ci_bounds) * 1.25)
    ax.set_yticks([1.0, 2.0, 4.0, 8.0])
    ax.set_yticklabels(["1x", "2x", "4x", "8x"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xticks(list(x))
    ax.set_xticklabels(XTICK_LABELS)
    ax.set_ylabel("Cost Reduction Factor", labelpad=4)
    ax.set_title("Algorithmic Progress")
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors="#333333", width=0.7, length=3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2 - bar.get_width() * 0.18,
            value * 1.08,
            f"{value:.1f}x",
            ha="right",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color="#333333",
        )

    fig.tight_layout(pad=0.4)
    fig.savefig(OUTPUT_PDF, format="pdf")
    fig.savefig(OUTPUT_PNG, format="png")
    return fig, ax


def main():
    open_license_estimates = load_open_license_estimates()
    algorithmic_progress = compute_algorithmic_progress(open_license_estimates)
    for benchmark, estimate in algorithmic_progress.items():
        lower, upper = estimate["ci"]
        print(
            f"{benchmark}: algorithmic={estimate['factor']:.3f}x "
            f"(90% CI [{lower:.3f}, {upper:.3f}])"
        )
    plot_algorithmic_progress(algorithmic_progress)


if __name__ == "__main__":
    main()
