"""
Price Threshold Analysis: Time Trends Below Price Thresholds

This script analyzes time trends for each benchmark when filtering models
below certain price thresholds (e.g., $1, $5, $10 benchmark cost).

Key question: Does progress look different when we only consider cheap models?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy import stats

sns.set_style("whitegrid")


def logit(p):
    """Convert probability to logit scale"""
    p_clipped = np.clip(p, 0.001, 0.999)
    return np.log(p_clipped / (1 - p_clipped))


def inverse_logit(logit_val):
    """Convert logit back to probability"""
    return 1 / (1 + np.exp(-logit_val))


def load_and_prepare_data(file_path, score_col, benchmark_name):
    """Load and prepare benchmark data"""
    df = pd.read_csv(file_path)
    df["Release Date"] = pd.to_datetime(df["Release Date"])

    # Clean score column
    df["Score"] = df[score_col].astype(str).str.replace("%", "").astype(float)

    # Clean price
    df["Price"] = (
        df["Benchmark Cost USD"].astype(str).str.replace("[$,]", "", regex=True)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Filter
    df_clean = df[["Model", "Release Date", "Score", "Price"]].dropna()
    df_clean = df_clean[(df_clean["Price"] > 0) & (df_clean["Score"] > 0)].copy()

    # Add transformations
    df_clean["log_Price"] = np.log10(df_clean["Price"])
    df_clean["Score_logit"] = logit(df_clean["Score"] / 100)

    # Add time variables
    min_date_ordinal = df_clean["Release Date"].min().toordinal()
    df_clean["Date_Ordinal"] = df_clean["Release Date"].map(datetime.toordinal)
    df_clean["Years_Since_Start"] = (
        df_clean["Date_Ordinal"] - min_date_ordinal
    ) / 365.25

    df_clean["Benchmark"] = benchmark_name

    return df_clean


def calculate_pareto_frontier(df):
    """
    Calculate Pareto frontier models (better OR cheaper than all previous).
    A model is on the Pareto frontier if it is not dominated by any other model.
    """
    df_work = df.copy().sort_values("Release Date")
    pareto_indices = []

    for date in df_work["Release Date"].unique():
        available_models = df_work[df_work["Release Date"] <= date].copy()
        available_models = available_models.sort_values(["Price", "Score"])

        frontier_indices = []

        for i, row in available_models.iterrows():
            dominated = False
            for j in frontier_indices:
                frontier_row = available_models.loc[j]
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
                frontier_indices.append(i)
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

        current_date_models = df_work[df_work["Release Date"] == date]
        for i, row in current_date_models.iterrows():
            if i in frontier_indices:
                pareto_indices.append(i)

    pareto_indices = list(set(pareto_indices))
    return df_work.loc[pareto_indices].copy()


def calculate_performance_frontier(df):
    """
    Calculate performance frontier models (strictly better than ALL previous models).
    Only includes models that beat the best previous performance, regardless of price.
    """
    df_sorted = df.copy().sort_values("Release Date")
    frontier_indices = []

    current_best = -np.inf

    for idx, row in df_sorted.iterrows():
        if row["Score_logit"] > current_best:
            frontier_indices.append(idx)
            current_best = row["Score_logit"]

    return df_sorted.loc[frontier_indices].copy()


def compute_best_under_budget_over_time(df, thresholds):
    """
    For each budget threshold B and each release date d, compute the best (max) score
    achievable using any model released up to d with Price <= B.

    Returns long-form DataFrame with one row per (B, d) where at least one model is feasible.
    """
    df_sorted = df.copy().sort_values("Release Date")
    dates = sorted(df_sorted["Release Date"].unique())

    rows = []
    for B in thresholds:
        for d in dates:
            avail = df_sorted[
                (df_sorted["Release Date"] <= d) & (df_sorted["Price"] <= B)
            ]
            if len(avail) == 0:
                continue
            best_idx = avail["Score_logit"].idxmax()
            best = avail.loc[best_idx]
            rows.append(
                {
                    "Benchmark": best["Benchmark"],
                    "Price Threshold": f"≤ ${B:.2f}",
                    "Threshold_Value": float(B),
                    "Release Date": d,
                    "Years_Since_Start": float(best["Years_Since_Start"]),
                    "Best Score (%)": float(best["Score"]),
                    "Best Score (logit)": float(best["Score_logit"]),
                    "N Available": int(len(avail)),
                    "Best Model": best["Model"],
                    "Best Model Price": float(best["Price"]),
                }
            )
    return pd.DataFrame(rows)


def run_regression_at_threshold(df, threshold, benchmark_name, sample_type="All"):
    """Run time trend regression for models below a price threshold"""
    df_filtered = df[df["Price"] <= threshold].copy()

    if len(df_filtered) < 3:
        return None

    X = df_filtered["Years_Since_Start"].values.reshape(-1, 1)
    y = df_filtered["Score_logit"].values

    model = LinearRegression().fit(X, y)
    r_squared = model.score(X, y)

    # Calculate p-value
    n = len(y)
    residuals = y - model.predict(X)
    mse = np.sum(residuals**2) / (n - 2)
    X_centered = X - np.mean(X)
    se = np.sqrt(mse / np.sum(X_centered**2))
    t_stat = model.coef_[0] / se
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))

    # Calculate 95% CI
    t_crit = stats.t.ppf(0.975, n - 2)
    ci_lower = model.coef_[0] - t_crit * se
    ci_upper = model.coef_[0] + t_crit * se

    # Calculate adjusted R²
    adj_r2 = 1 - (1 - r_squared) * (n - 1) / (n - 2)

    return {
        "Benchmark": benchmark_name,
        "Sample Type": sample_type,
        "Price Threshold": f"≤ ${threshold:.2f}",
        "N": n,
        "Time Coef (logits/yr)": model.coef_[0],
        "Std Error": se,
        "p-value": p_value,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper,
        "R²": r_squared,
        "Adj R²": adj_r2,
        "Mean Score (%)": df_filtered["Score"].mean(),
        "Score Range": f"{df_filtered['Score'].min():.1f}-{df_filtered['Score'].max():.1f}%",
    }


def main():
    print("=" * 100)
    print("PRICE THRESHOLD ANALYSIS: TIME TRENDS FOR LOW-COST MODELS")
    print("=" * 100)
    print()

    # Load data
    print("Loading benchmark data...")
    df_gpqa = load_and_prepare_data(
        "data/price_reduction_models.csv", "epoch_gpqa", "GPQA-D"
    )
    df_swe = load_and_prepare_data(
        "data/swe_price_reduction_models.csv", "epoch_swe", "SWE-Bench"
    )
    df_aime = load_and_prepare_data(
        "data/aime_price_reduction_models.csv", "oneshot_AIME", "AIME"
    )

    print(f"GPQA-D: {len(df_gpqa)} models")
    print(f"SWE-Bench: {len(df_swe)} models")
    print(f"AIME: {len(df_aime)} models")
    print()

    # Define price thresholds to test
    thresholds = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    results = []

    # Helper function to run regression on a dataset
    def run_regression_helper(df, name, sample_type, threshold_label):
        if len(df) < 3:
            return None
        X = df["Years_Since_Start"].values.reshape(-1, 1)
        y = df["Score_logit"].values
        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)
        n = len(y)
        residuals = y - model.predict(X)
        mse = np.sum(residuals**2) / (n - 2)
        X_centered = X - np.mean(X)
        se = np.sqrt(mse / np.sum(X_centered**2))
        t_stat = model.coef_[0] / se
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
        t_crit = stats.t.ppf(0.975, n - 2)
        ci_lower = model.coef_[0] - t_crit * se
        ci_upper = model.coef_[0] + t_crit * se
        adj_r2 = 1 - (1 - r_squared) * (n - 1) / (n - 2)

        return {
            "Benchmark": name,
            "Sample Type": sample_type,
            "Price Threshold": threshold_label,
            "N": n,
            "Time Coef (logits/yr)": model.coef_[0],
            "Std Error": se,
            "p-value": p_value,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper,
            "R²": r_squared,
            "Adj R²": adj_r2,
            "Mean Score (%)": df["Score"].mean(),
            "Score Range": f"{df['Score'].min():.1f}-{df['Score'].max():.1f}%",
        }

    # Add full sample results for all three sample types
    for df, name in [(df_gpqa, "GPQA-D"), (df_swe, "SWE-Bench"), (df_aime, "AIME")]:
        # All models
        result = run_regression_helper(df, name, "All", "All models")
        if result:
            results.append(result)

        # Pareto frontier
        df_pareto = calculate_pareto_frontier(df)
        result = run_regression_helper(df_pareto, name, "Pareto", "All models")
        if result:
            results.append(result)

        # Performance frontier
        df_frontier = calculate_performance_frontier(df)
        result = run_regression_helper(df_frontier, name, "Frontier", "All models")
        if result:
            results.append(result)

    # Run analysis for each benchmark, sample type, and threshold
    for df, name in [(df_gpqa, "GPQA-D"), (df_swe, "SWE-Bench"), (df_aime, "AIME")]:
        for threshold in thresholds:
            # All models below threshold
            result = run_regression_at_threshold(df, threshold, name, "All")
            if result is not None:
                results.append(result)

            # Pareto frontier below threshold
            df_filtered = df[df["Price"] <= threshold].copy()
            if len(df_filtered) >= 3:
                df_pareto = calculate_pareto_frontier(df_filtered)
                result = run_regression_helper(
                    df_pareto, name, "Pareto", f"≤ ${threshold:.2f}"
                )
                if result:
                    results.append(result)

            # Performance frontier below threshold
            if len(df_filtered) >= 3:
                df_frontier = calculate_performance_frontier(df_filtered)
                result = run_regression_helper(
                    df_frontier, name, "Frontier", f"≤ ${threshold:.2f}"
                )
                if result:
                    results.append(result)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Print summary table
    print("\n" + "=" * 120)
    print("TIME TREND REGRESSIONS AT DIFFERENT PRICE THRESHOLDS")
    print("Model: logit(score) ~ years_since_start")
    print("Comparing: All models vs Pareto (better OR cheaper) vs Frontier (best ever)")
    print("=" * 120 + "\n")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 4)

    # Sort for better readability
    results_df_sorted = results_df.sort_values(
        ["Benchmark", "Sample Type", "Price Threshold"]
    )
    print(results_df_sorted.to_string(index=False))

    # Save results
    results_df.to_csv("results/price_threshold_regressions.csv", index=False)
    print("\n\nSaved to: results/price_threshold_regressions.csv")

    # -------------------------------------------------------------------------
    # Best performance achievable within a budget (what you said you care about)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("BEST PERFORMANCE ACHIEVABLE WITHIN A BUDGET (STEP-FUNCTION OVER TIME)")
    print(
        "For each budget B and date d: max score among models released up to d with Price <= B"
    )
    print(
        "Regression: best_logit_score ~ years_since_start (one point per date per budget)"
    )
    print("=" * 120 + "\n")

    best_budget_series_all = []
    best_budget_reg_rows = []

    for df, name in [(df_gpqa, "GPQA-D"), (df_swe, "SWE-Bench"), (df_aime, "AIME")]:
        best_df = compute_best_under_budget_over_time(df, thresholds)
        if len(best_df) == 0:
            continue
        best_budget_series_all.append(best_df)

        for B in thresholds:
            sub = best_df[best_df["Threshold_Value"] == float(B)].copy()
            if len(sub) < 3:
                continue

            X = sub["Years_Since_Start"].values.reshape(-1, 1)
            y = sub["Best Score (logit)"].values

            model = LinearRegression().fit(X, y)
            r_squared = model.score(X, y)

            n = len(y)
            residuals = y - model.predict(X)
            mse = np.sum(residuals**2) / (n - 2)
            X_centered = X - np.mean(X)
            se = np.sqrt(mse / np.sum(X_centered**2))
            t_stat = model.coef_[0] / se
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
            t_crit = stats.t.ppf(0.975, n - 2)
            ci_lower = model.coef_[0] - t_crit * se
            ci_upper = model.coef_[0] + t_crit * se

            last_best_score = sub.sort_values("Release Date")["Best Score (%)"].iloc[-1]

            best_budget_reg_rows.append(
                {
                    "Benchmark": name,
                    "Budget Threshold": f"≤ ${B:.2f}",
                    "Threshold_Value": float(B),
                    "N_dates": int(n),
                    "Time Coef (logits/yr)": float(model.coef_[0]),
                    "Std Error": float(se),
                    "p-value": float(p_value),
                    "95% CI Lower": float(ci_lower),
                    "95% CI Upper": float(ci_upper),
                    "R²": float(r_squared),
                    "Mean Best Score (%)": float(sub["Best Score (%)"].mean()),
                    "Last Best Score (%)": float(last_best_score),
                    "Max Best Score (%)": float(sub["Best Score (%)"].max()),
                }
            )

    best_budget_reg_df = pd.DataFrame(best_budget_reg_rows)
    if len(best_budget_reg_df) > 0:
        best_budget_reg_df.to_csv(
            "results/best_under_budget_time_trends.csv", index=False
        )
        print("Saved to: results/best_under_budget_time_trends.csv\n")
        print(
            best_budget_reg_df.sort_values(["Benchmark", "Threshold_Value"]).to_string(
                index=False
            )
        )

        # Plot: slope vs budget (best-under-budget step function)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, benchmark in enumerate(["GPQA-D", "SWE-Bench", "AIME"]):
            ax = axes[idx]
            sub = best_budget_reg_df[
                best_budget_reg_df["Benchmark"] == benchmark
            ].copy()
            if len(sub) == 0:
                ax.set_title(f"{benchmark} (no data)")
                ax.axis("off")
                continue
            sub = sub.sort_values("Threshold_Value")
            ax.plot(
                sub["Threshold_Value"],
                sub["Time Coef (logits/yr)"],
                "o-",
                linewidth=2,
            )
            ax.fill_between(
                sub["Threshold_Value"],
                sub["95% CI Lower"],
                sub["95% CI Upper"],
                alpha=0.2,
            )
            ax.set_xscale("log")
            ax.set_xlabel("Budget Threshold (USD)", fontsize=11, fontweight="bold")
            ax.set_ylabel(
                "Best-under-budget time coef (logits/yr)",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_title(
                f"{benchmark}\nBest achievable under budget: slope vs budget",
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            for _, row in sub.iterrows():
                ax.annotate(
                    f"n={int(row['N_dates'])}",
                    xy=(row["Threshold_Value"], row["Time Coef (logits/yr)"]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=7,
                    ha="center",
                    alpha=0.7,
                )

        plt.tight_layout()
        plt.savefig(
            "figures/best_under_budget_time_trends.png", dpi=300, bbox_inches="tight"
        )
        print("\nSaved: figures/best_under_budget_time_trends.png")
        plt.show()

        # Plot: GPQA-D best-under-fixed-budget curves over time (requested)
        gpqa_budgets = [0.01, 0.05, 0.1, 1.0, 10.0, 50.0]
        if len(best_budget_series_all) > 0:
            all_series = pd.concat(best_budget_series_all, ignore_index=True)
            gpqa_series = all_series[all_series["Benchmark"] == "GPQA-D"].copy()
            if len(gpqa_series) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                # Add unconstrained best-achievable step curve (no budget limit)
                # Using the same dates as the budget series: for each date, best score among all models released up to that date.
                df_gpqa_sorted = df_gpqa.sort_values("Release Date").copy()
                unconstrained_rows = []
                for d in sorted(df_gpqa_sorted["Release Date"].unique()):
                    avail = df_gpqa_sorted[df_gpqa_sorted["Release Date"] <= d]
                    if len(avail) == 0:
                        continue
                    best_idx = avail["Score_logit"].idxmax()
                    best = avail.loc[best_idx]
                    unconstrained_rows.append(
                        {"Release Date": d, "Best Score (%)": float(best["Score"])}
                    )
                if len(unconstrained_rows) > 0:
                    unconstrained = pd.DataFrame(unconstrained_rows).sort_values(
                        "Release Date"
                    )
                    ax.step(
                        unconstrained["Release Date"],
                        unconstrained["Best Score (%)"],
                        where="post",
                        linewidth=2.5,
                        linestyle="--",
                        color="black",
                        label="No budget",
                    )

                for B in gpqa_budgets:
                    sub = gpqa_series[gpqa_series["Threshold_Value"] == float(B)].copy()
                    if len(sub) == 0:
                        continue
                    sub = sub.sort_values("Release Date")
                    ax.step(
                        sub["Release Date"],
                        sub["Best Score (%)"],
                        where="post",
                        linewidth=2,
                        label=f"≤ ${B:g}",
                    )

                ax.set_title(
                    "GPQA-D: Best achievable score under fixed budget over time",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_xlabel("Release Date", fontsize=11, fontweight="bold")
                ax.set_ylabel("Best Score (%)", fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.legend(title="Budget", fontsize=9)
                plt.tight_layout()
                plt.savefig(
                    "figures/gpqa_best_under_budget_curves.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                print("Saved: figures/gpqa_best_under_budget_curves.png")
                plt.show()
    else:
        print(
            "No best-under-budget regressions had >= 3 date points (try higher budgets)."
        )

    # Create visualization
    print("\nCreating visualizations...")

    # Plot 1: Time coefficient vs price threshold
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    benchmarks = ["GPQA-D", "SWE-Bench", "AIME"]
    colors = ["red", "blue", "green"]

    for idx, (benchmark, color) in enumerate(zip(benchmarks, colors)):
        ax = axes[idx]

        # Get data for this benchmark (All models only for this plot)
        bench_data = results_df[
            (results_df["Benchmark"] == benchmark)
            & (results_df["Sample Type"] == "All")
        ].copy()

        # Separate full sample from thresholded samples
        full_sample = bench_data[bench_data["Price Threshold"] == "All models"]
        threshold_data = bench_data[
            bench_data["Price Threshold"] != "All models"
        ].copy()

        if len(threshold_data) == 0:
            continue

        # Extract numeric threshold values
        threshold_data["Threshold_Value"] = (
            threshold_data["Price Threshold"].str.extract(r"(\d+\.?\d*)").astype(float)
        )
        threshold_data = threshold_data.sort_values("Threshold_Value")

        # Plot
        ax.plot(
            threshold_data["Threshold_Value"],
            threshold_data["Time Coef (logits/yr)"],
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label="Thresholded samples",
        )

        # Add horizontal line for full sample
        if len(full_sample) > 0:
            full_coef = full_sample["Time Coef (logits/yr)"].values[0]
            ax.axhline(
                y=full_coef,
                color=color,
                linestyle="--",
                linewidth=2,
                alpha=0.5,
                label=f"Full sample ({full_coef:.2f})",
            )

        # Add confidence intervals
        ax.fill_between(
            threshold_data["Threshold_Value"],
            threshold_data["95% CI Lower"],
            threshold_data["95% CI Upper"],
            alpha=0.2,
            color=color,
        )

        ax.set_xscale("log")
        ax.set_xlabel("Price Threshold (USD)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Time Coefficient (logits/yr)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{benchmark}\nTime Trend vs Price Threshold",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add sample size annotations
        for _, row in threshold_data.iterrows():
            ax.annotate(
                f"N={row['N']}",
                xy=(row["Threshold_Value"], row["Time Coef (logits/yr)"]),
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                alpha=0.7,
            )

    plt.tight_layout()
    plt.savefig("figures/price_threshold_time_trends.png", dpi=300, bbox_inches="tight")
    print("Saved: figures/price_threshold_time_trends.png")
    plt.show()

    # Plot 2: Sample size vs threshold (by sample type)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sample_types = ["All", "Pareto", "Frontier"]
    sample_colors = {"All": "#2C3E50", "Pareto": "#E74C3C", "Frontier": "#27AE60"}
    sample_labels = {
        "All": "All models",
        "Pareto": "Pareto (better OR cheaper)",
        "Frontier": "Frontier (best ever)",
    }

    for idx, (benchmark, base_color) in enumerate(zip(benchmarks, colors)):
        ax = axes[idx]

        bench_data = results_df[results_df["Benchmark"] == benchmark].copy()

        for sample_type in sample_types:
            sample_data = bench_data[bench_data["Sample Type"] == sample_type].copy()
            threshold_data = sample_data[
                sample_data["Price Threshold"] != "All models"
            ].copy()

            if len(threshold_data) == 0:
                continue

            threshold_data["Threshold_Value"] = (
                threshold_data["Price Threshold"]
                .str.extract(r"(\d+\.?\d*)")
                .astype(float)
            )
            threshold_data = threshold_data.sort_values("Threshold_Value")

            ax.plot(
                threshold_data["Threshold_Value"],
                threshold_data["N"],
                "o-",
                color=sample_colors[sample_type],
                linewidth=2,
                markersize=6,
                label=sample_labels[sample_type],
                alpha=0.8,
            )

        # Add horizontal lines for full sample sizes
        for sample_type in sample_types:
            sample_data = bench_data[bench_data["Sample Type"] == sample_type].copy()
            full_sample = sample_data[sample_data["Price Threshold"] == "All models"]
            if len(full_sample) > 0:
                full_n = full_sample["N"].values[0]
                ax.axhline(
                    y=full_n,
                    color=sample_colors[sample_type],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.3,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Price Threshold (USD)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Number of Models", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{benchmark}\nSample Size vs Price Threshold",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "figures/price_threshold_sample_sizes.png", dpi=300, bbox_inches="tight"
    )
    print("Saved: figures/price_threshold_sample_sizes.png")
    plt.show()

    # Plot 3: Time coefficient by sample type (All vs Pareto vs Frontier)
    print("\nCreating sample type comparison plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (benchmark, base_color) in enumerate(zip(benchmarks, colors)):
        ax = axes[idx]

        # Get data for this benchmark
        bench_data = results_df[results_df["Benchmark"] == benchmark].copy()

        for sample_type in sample_types:
            sample_data = bench_data[bench_data["Sample Type"] == sample_type].copy()

            # Separate full sample from thresholded samples
            threshold_data = sample_data[
                sample_data["Price Threshold"] != "All models"
            ].copy()

            if len(threshold_data) == 0:
                continue

            # Extract numeric threshold values
            threshold_data["Threshold_Value"] = (
                threshold_data["Price Threshold"]
                .str.extract(r"(\d+\.?\d*)")
                .astype(float)
            )
            threshold_data = threshold_data.sort_values("Threshold_Value")

            # Plot
            ax.plot(
                threshold_data["Threshold_Value"],
                threshold_data["Time Coef (logits/yr)"],
                "o-",
                color=sample_colors[sample_type],
                linewidth=2,
                markersize=6,
                label=sample_labels[sample_type],
                alpha=0.8,
            )

        # Add horizontal lines for full samples
        for sample_type in sample_types:
            sample_data = bench_data[bench_data["Sample Type"] == sample_type].copy()
            full_sample = sample_data[sample_data["Price Threshold"] == "All models"]
            if len(full_sample) > 0:
                full_coef = full_sample["Time Coef (logits/yr)"].values[0]
                ax.axhline(
                    y=full_coef,
                    color=sample_colors[sample_type],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.3,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Price Threshold (USD)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Time Coefficient (logits/yr)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{benchmark}\nAll vs Pareto vs Frontier", fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "figures/price_threshold_sample_type_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Saved: figures/price_threshold_sample_type_comparison.png")
    plt.show()

    print("\nAnalysis complete!")
    print("\nKEY INSIGHTS:")
    print("=" * 100)
    print(
        "Plot 1 (Time Trends): Compare time coefficients at different price thresholds"
    )
    print(
        "  - Lower thresholds = only cheap models, higher thresholds = include expensive models"
    )
    print(
        "  - If coefficient increases with threshold, expensive models drive progress"
    )
    print("  - If coefficient stays flat, progress is not price-dependent")
    print()
    print(
        "Plot 2 (Sample Sizes): Shows how many models are available at each threshold"
    )
    print()
    print("Plot 3 (Sample Type Comparison): All vs Pareto vs Frontier")
    print("  - All models: Every model below threshold")
    print("  - Pareto: Models better OR cheaper than all previous (not dominated)")
    print("  - Frontier: Models strictly better than ALL previous (best ever)")
    print("  - Compare how these three groups show different progress patterns")
    print("=" * 100)


if __name__ == "__main__":
    main()
