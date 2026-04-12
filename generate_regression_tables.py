#!/usr/bin/env python3
"""
Generate regression comparison tables (CSV + LaTeX).

Extracts the regression pipeline from main_regresssion.ipynb so it can run
headless (no tkinter, no notebook kernel needed).

Outputs:
    results_data/regression_comparison_table_with_arc.csv
    results_data/regression_comparison_table_with_arc.tex
"""

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from datetime import datetime
from scipy import stats


# ---------------------------------------------------------------------------
# Core regression function (from main_regresssion.ipynb cell 1)
# ---------------------------------------------------------------------------
def plot_price_mmlu_regression(
    df,
    open_license_only=False,
    min_mmlu=40,
    max_mmlu=70,
    price_column="Output Price\nUSD/1M Tokens",
    exclude_dominated=False,
    benchmark_col="MMLU-Pro (Reasoning & Knowledge)",
    exclude_reasoning=False,
    use_huber=False,
    huber_epsilon=1.35,
    huber_max_iter=100,
    pareto_frontier_only=False,
    use_logit=False,
    show_plot=False,
    show_model_names=False,
    year_filter=None,
):
    mmlu_col = benchmark_col
    price_col = price_column
    license_col = "License"
    reasoning_col = "Reasoning_TF"

    df_work = df.copy()

    df_work[mmlu_col] = (
        df_work[mmlu_col].astype(str).str.replace("%", "", regex=False).astype(float)
    )

    if use_logit:
        proportions = df_work[mmlu_col] / 100.0
        proportions = np.clip(proportions, 1e-10, 1 - 1e-10)
        df_work[f"{mmlu_col}_logit"] = np.log(proportions / (1 - proportions))
        mmlu_col_transformed = f"{mmlu_col}_logit"
    else:
        mmlu_col_transformed = mmlu_col

    df_work[price_col] = (
        df_work[price_col].astype(str).str.replace("[$,]", "", regex=True)
    )
    df_work[price_col] = pd.to_numeric(df_work[price_col], errors="coerce")

    if open_license_only:
        df_work = df_work[
            df_work[license_col].notna()
            & df_work[license_col].str.contains("open", case=False, na=False)
        ]

    if exclude_reasoning and reasoning_col in df_work.columns:
        df_work = df_work[df_work[reasoning_col] != True]

    df_sub = df_work.dropna(subset=["Release Date", price_col, mmlu_col])
    df_sub = df_sub[(df_sub[price_col] > 0) & (df_sub[mmlu_col] > 0)]

    if year_filter is not None:
        if not pd.api.types.is_datetime64_any_dtype(df_sub["Release Date"]):
            df_sub["Release Date"] = pd.to_datetime(df_sub["Release Date"])
        df_sub = df_sub[df_sub["Release Date"].dt.year == year_filter]

    df_sub = df_sub[(df_sub[mmlu_col] >= min_mmlu) & (df_sub[mmlu_col] <= max_mmlu)]

    df_sub_display = df_sub.copy()
    if exclude_dominated:
        df_sub_display = df_sub_display.sort_values("Release Date")
        non_dominated = []
        for i, row in df_sub_display.iterrows():
            dominated = False
            for j in non_dominated:
                prev_row = df_sub_display.loc[j]
                if (
                    prev_row[mmlu_col] >= row[mmlu_col]
                    and prev_row[price_col] <= row[price_col]
                    and (
                        prev_row[mmlu_col] > row[mmlu_col]
                        or prev_row[price_col] < row[price_col]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(i)
                new_non_dominated = []
                for j in non_dominated[:-1]:
                    prev_row = df_sub_display.loc[j]
                    if not (
                        row[mmlu_col] >= prev_row[mmlu_col]
                        and row[price_col] <= prev_row[price_col]
                        and (
                            row[mmlu_col] > prev_row[mmlu_col]
                            or row[price_col] < prev_row[price_col]
                        )
                    ):
                        new_non_dominated.append(j)
                non_dominated = new_non_dominated + [i]
        df_sub_display = df_sub_display.loc[non_dominated]

    if pareto_frontier_only:
        df_regression = df_sub.sort_values("Release Date").copy()
        pareto_indices = []
        for date in df_regression["Release Date"].unique():
            available_models = df_regression[
                df_regression["Release Date"] <= date
            ].copy()
            available_models = available_models.sort_values([price_col, mmlu_col])
            frontier_indices = []
            for i, row in available_models.iterrows():
                dominated = False
                for j in frontier_indices:
                    frontier_row = available_models.loc[j]
                    if (
                        frontier_row[mmlu_col] >= row[mmlu_col]
                        and frontier_row[price_col] <= row[price_col]
                        and (
                            frontier_row[mmlu_col] > row[mmlu_col]
                            or frontier_row[price_col] < row[price_col]
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
                            row[mmlu_col] >= frontier_row[mmlu_col]
                            and row[price_col] <= frontier_row[price_col]
                            and (
                                row[mmlu_col] > frontier_row[mmlu_col]
                                or row[price_col] < frontier_row[price_col]
                            )
                        ):
                            new_frontier_indices.append(j)
                    frontier_indices = new_frontier_indices + [i]
            current_date_models = df_regression[df_regression["Release Date"] == date]
            for i, row in current_date_models.iterrows():
                if i in frontier_indices:
                    pareto_indices.append(i)
        pareto_indices = list(set(pareto_indices))
        df_regression = df_regression.loc[pareto_indices]
    else:
        df_regression = df_sub.copy()

    if len(df_regression) < 3:
        print(
            f"Warning: Only {len(df_regression)} data points available for regression. Need at least 3."
        )
        return None, None, None

    df_regression = df_regression.sort_values("Release Date").copy()
    df_regression["Date_Ordinal"] = df_regression["Release Date"].map(
        datetime.toordinal
    )

    X = np.column_stack(
        [
            df_regression["Date_Ordinal"].values,
            df_regression[mmlu_col_transformed].values,
        ]
    )
    y = np.log(df_regression[price_col].values)

    if use_huber:
        model = HuberRegressor(epsilon=huber_epsilon, max_iter=huber_max_iter).fit(X, y)
        alpha, beta = model.coef_
        c = model.intercept_
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        reg_type = "Huber"
    else:
        model = LinearRegression().fit(X, y)
        alpha, beta = model.coef_
        c = model.intercept_
        y_pred = model.predict(X)
        r2 = model.score(X, y)
        reg_type = "OLS"

    annual_log_change = alpha * 365
    annual_pct_change = (np.exp(annual_log_change) - 1) * 100
    factor_change_per_year = np.exp(annual_log_change)
    factor_decrease_per_year = 1 / factor_change_per_year

    if not use_huber:
        n = len(df_regression)
        p = 2
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - p - 1)
        X_mean_centered = X - np.mean(X, axis=0)
        cov_matrix = np.linalg.inv(X_mean_centered.T.dot(X_mean_centered)) * mse
        se_alpha = np.sqrt(cov_matrix[0, 0])
        se_annual = se_alpha * 365
        t_stat = stats.t.ppf(0.95, n - p - 1)
        annual_log_change_lower = annual_log_change - t_stat * se_annual
        annual_log_change_upper = annual_log_change + t_stat * se_annual
        factor_change_lower = np.exp(annual_log_change_lower)
        factor_change_upper = np.exp(annual_log_change_upper)
        factor_decrease_lower = 1 / factor_change_upper
        factor_decrease_upper = 1 / factor_change_lower
    else:
        factor_change_lower = None
        factor_change_upper = None
        factor_decrease_lower = None
        factor_decrease_upper = None

    benchmark_name = benchmark_col.split(" (")[0]
    transform_desc = "logit" if use_logit else "linear"
    data_source = "Pareto frontier only" if pareto_frontier_only else "all data"

    print(f"\nRegression Results ({reg_type}):")
    print(f"Data used: {data_source}")
    print(
        f"Model: log(Price) = {alpha:.6f}*time + {beta:.3f}*{benchmark_name}({transform_desc}) + {c:.3f}"
    )
    print(f"R² score: {r2:.4f}")
    print(f"Annual factor decrease: {factor_decrease_per_year:.3f}x/yr")
    if factor_decrease_lower is not None:
        print(
            f"90% CI for factor decrease: [{factor_decrease_lower:.3f}, {factor_decrease_upper:.3f}]"
        )
    print(f"Data points used for regression: {len(df_regression)}")

    return (
        model,
        df_regression,
        {
            "alpha": alpha,
            "beta": beta,
            "c": c,
            "annual_pct_change": annual_pct_change,
            "factor_change_per_year": factor_change_per_year,
            "factor_decrease_per_year": factor_decrease_per_year,
            "factor_change_ci_lower": factor_change_lower,
            "factor_change_ci_upper": factor_change_upper,
            "factor_decrease_ci_lower": factor_decrease_lower,
            "factor_decrease_ci_upper": factor_decrease_upper,
            "r2_score": r2,
            "regression_type": reg_type,
            "pareto_frontier_only": pareto_frontier_only,
        },
    )


# ---------------------------------------------------------------------------
# Table generation (from main_regresssion.ipynb cells 8 + 15)
# ---------------------------------------------------------------------------
def create_arc_comparison_table(
    df_gpqa, df_aime, df_swe, df_arc_clean,
    hardware_gain_factor=1.0, year_filter=None,
):
    configurations = [
        {"name": "Pareto_restricted_all_license", "pareto_frontier_only": True, "open_license_only": False},
        {"name": "pareto_restricted_open_license", "pareto_frontier_only": True, "open_license_only": True},
        {"name": "all_license_no_restriction", "pareto_frontier_only": False, "open_license_only": False},
        {"name": "open_license_only_no_restriction", "pareto_frontier_only": False, "open_license_only": True},
    ]

    benchmarks = [
        {"name": "GPQA", "df": df_gpqa, "benchmark_col": "epoch_gpqa", "price_col": "Benchmark Cost USD", "min_mmlu": 0, "max_mmlu": 100, "use_logit": True},
        {"name": "AIME", "df": df_aime, "benchmark_col": "oneshot_AIME", "price_col": "Benchmark Cost USD", "min_mmlu": 0, "max_mmlu": 100, "use_logit": True},
        {"name": "SWE-Bench", "df": df_swe, "benchmark_col": "epoch_swe", "price_col": "Benchmark Cost USD", "min_mmlu": 0, "max_mmlu": 100, "use_logit": True},
        {"name": "ARC-AGI", "df": df_arc_clean, "benchmark_col": "arc_score_clean", "price_col": "arc_price_clean", "min_mmlu": 0, "max_mmlu": 100, "use_logit": True},
    ]

    results_data = []
    for benchmark in benchmarks:
        for config in configurations:
            print(f"\nProcessing {benchmark['name']} - {config['name']}...")
            model, data, results = plot_price_mmlu_regression(
                df=benchmark["df"],
                open_license_only=config["open_license_only"],
                price_column=benchmark["price_col"],
                exclude_dominated=False,
                benchmark_col=benchmark["benchmark_col"],
                min_mmlu=benchmark["min_mmlu"],
                max_mmlu=benchmark["max_mmlu"],
                exclude_reasoning=False,
                use_huber=False,
                pareto_frontier_only=config["pareto_frontier_only"],
                use_logit=benchmark["use_logit"],
                show_plot=False,
                year_filter=year_filter,
            )

            if results is not None:
                factor_decrease = results["factor_decrease_per_year"]
                ci_lower = results["factor_decrease_ci_lower"]
                ci_upper = results["factor_decrease_ci_upper"]
                r2 = results["r2_score"]
                n = len(data) if data is not None else 0
                factor_decrease_adjusted = factor_decrease / hardware_gain_factor
                if ci_lower is not None and ci_upper is not None:
                    ci_lower_adjusted = ci_lower / hardware_gain_factor
                    ci_upper_adjusted = ci_upper / hardware_gain_factor
                    ci_str = f"[{ci_lower_adjusted:.3f}, {ci_upper_adjusted:.3f}]"
                else:
                    ci_str = "N/A"
                results_data.append({
                    "Benchmark": benchmark["name"],
                    "Configuration": config["name"],
                    "Annual Factor Decrease": f"{factor_decrease_adjusted:.3f}",
                    "90% CI": ci_str,
                    "n": n,
                    "R²": f"{r2:.4f}",
                })
            else:
                results_data.append({
                    "Benchmark": benchmark["name"],
                    "Configuration": config["name"],
                    "Annual Factor Decrease": "N/A",
                    "90% CI": "N/A",
                    "n": 0,
                    "R²": "N/A",
                })

    results_df = pd.DataFrame(results_data)
    results_df.columns = ["Benchmark", "Restriction", "Year Decrease Factor", "90% CI", "n", "R²"]

    benchmark_order = ["GPQA", "AIME", "SWE-Bench", "ARC-AGI"]
    config_order = [
        "Pareto_restricted_all_license",
        "pareto_restricted_open_license",
        "all_license_no_restriction",
        "open_license_only_no_restriction",
    ]
    ordered_rows = []
    for bm in benchmark_order:
        for cfg in config_order:
            row = results_df[(results_df["Benchmark"] == bm) & (results_df["Restriction"] == cfg)]
            if not row.empty:
                ordered_rows.append(row)
    results_df = pd.concat(ordered_rows, ignore_index=True)

    restriction_mapping = {
        "Pareto_restricted_all_license": "Pareto Restricted All License",
        "pareto_restricted_open_license": "Pareto Restricted Open License",
        "all_license_no_restriction": "All License (no restriction)",
        "open_license_only_no_restriction": "Open License (no restriction)",
    }
    results_df["Restriction"] = results_df["Restriction"].map(restriction_mapping)

    for i in range(1, len(results_df)):
        if results_df.loc[i, "Benchmark"] == results_df.loc[i - 1, "Benchmark"]:
            results_df.loc[i, "Benchmark"] = ""

    return results_df


def _save_latex(table, path, caption, label):
    """Save a DataFrame as a LaTeX table, with jinja2 fallback."""
    try:
        latex_table = table.to_latex(
            index=False, escape=False,
            column_format="|l|l|c|c|c|c|",
            caption=caption, label=label,
        )
    except ImportError:
        lines = [
            r"\begin{table}", r"\centering",
            rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            r"\begin{tabular}{|l|l|c|c|c|c|}", r"\hline",
            " & ".join(table.columns) + r" \\", r"\hline",
        ]
        for _, row in table.iterrows():
            lines.append(" & ".join(str(v) for v in row) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        latex_table = "\n".join(lines)
    with open(path, "w") as f:
        f.write(latex_table)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load data (mirrors notebook cells 2, 3, 5, 11)
    print("Loading data...")

    df_gpqa = pd.read_csv("data/gpqa_price_reduction_models.csv")
    df_gpqa["Release Date"] = pd.to_datetime(df_gpqa["Release Date"])
    df_gpqa["Active Parameters"] = np.where(
        df_gpqa["Known Active Parameters"].notna(),
        df_gpqa["Known Active Parameters"],
        df_gpqa["Parameters"],
    )

    df_swe = pd.read_csv("data/swe_price_reduction_models.csv")
    df_swe["Release Date"] = pd.to_datetime(df_swe["Release Date"])
    df_swe["Active Parameters"] = np.where(
        df_swe["Known Active Parameters"].notna(),
        df_swe["Known Active Parameters"],
        df_swe["Parameters"],
    )

    df_aime = pd.read_csv("data/aime_price_reduction_models.csv")
    df_aime["Release Date"] = pd.to_datetime(df_aime["Release Date"])
    df_aime["Active Parameters"] = np.where(
        df_aime["Known Active Parameters"].notna(),
        df_aime["Known Active Parameters"],
        df_aime["Parameters"],
    )

    df_arc = pd.read_csv("data/merged.csv")
    df_arc["Release Date"] = pd.to_datetime(df_arc["Release Date"], errors="coerce")
    df_arc["arc_score_clean"] = (
        df_arc["arc_ARC-AGI-1"].astype(str).str.replace("%", "").astype(float)
    )
    df_arc["arc_price_clean"] = (
        df_arc["arc_Cost/Task"].astype(str).str.replace("$", "").str.replace(",", "")
    )
    df_arc["arc_price_clean"] = pd.to_numeric(df_arc["arc_price_clean"], errors="coerce")
    df_arc_clean = df_arc[
        df_arc["arc_score_clean"].notna()
        & df_arc["Release Date"].notna()
        & df_arc["arc_price_clean"].notna()
    ].copy()
    df_arc_clean = df_arc_clean[df_arc_clean["arc_price_clean"] > 0]
    print(f"Loaded: GPQA={len(df_gpqa)}, AIME={len(df_aime)}, SWE={len(df_swe)}, ARC={len(df_arc_clean)}")

    # ── Raw table (no hardware adjustment) ──
    print("\n" + "=" * 80)
    print("REGRESSION TABLE — RAW (no hardware adjustment)")
    print("=" * 80)

    table_raw = create_arc_comparison_table(
        df_gpqa, df_aime, df_swe, df_arc_clean,
        hardware_gain_factor=1.0,
    )
    print("\n")
    print(table_raw.to_string(index=False))

    table_raw.to_csv("results_data/regression_comparison_table_raw.csv", index=False)
    print("\nSaved results_data/regression_comparison_table_raw.csv")
    _save_latex(table_raw, "results_data/regression_comparison_table_raw.tex",
                caption="Regression Results (Raw, No Hardware Adjustment)",
                label="tab:regression_results_raw")

    # ── Hardware-adjusted table (divide by 1/0.7 ≈ 1.43) ──
    print("\n" + "=" * 80)
    print("REGRESSION TABLE — HARDWARE-ADJUSTED (÷ 1/0.7)")
    print("=" * 80)

    table_hw = create_arc_comparison_table(
        df_gpqa, df_aime, df_swe, df_arc_clean,
        hardware_gain_factor=(1 / 0.7),
    )
    print("\n")
    print(table_hw.to_string(index=False))

    table_hw.to_csv("results_data/regression_comparison_table_hw_adjusted.csv", index=False)
    print("\nSaved results_data/regression_comparison_table_hw_adjusted.csv")
    _save_latex(table_hw, "results_data/regression_comparison_table_hw_adjusted.tex",
                caption="Regression Results (Hardware-Adjusted, ÷ 1/0.7)",
                label="tab:regression_results_hw_adjusted")
