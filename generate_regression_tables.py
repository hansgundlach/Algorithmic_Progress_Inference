#!/usr/bin/env python3
"""
Generate regression comparison tables (CSV + LaTeX).

Extracts the regression pipeline from main_regresssion.ipynb so it can run
headless (no tkinter, no notebook kernel needed).

Outputs:
    results_data/regression_comparison_table_raw.csv
    results_data/regression_comparison_table_hw_adjusted.csv
    results_data/regression_comparison_table.csv
    results_data/*.tex
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
        # Cap at [0.5%, 99.5%] to prevent extreme logit values from
        # dominating the regression (logit is undefined at 0 and 1).
        proportions = np.clip(proportions, 0.005, 0.995)
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
def create_comparison_table(
    df_gpqa, df_aime, df_swe, df_arc_clean=None,
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
    ]
    if df_arc_clean is not None and not df_arc_clean.empty:
        benchmarks.append(
            {"name": "ARC-AGI", "df": df_arc_clean, "benchmark_col": "arc_score_clean", "price_col": "arc_price_clean", "min_mmlu": 0, "max_mmlu": 100, "use_logit": True}
        )

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

    benchmark_order = ["GPQA", "AIME", "SWE-Bench"]
    if df_arc_clean is not None and not df_arc_clean.empty:
        benchmark_order.append("ARC-AGI")
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


# Backward-compatible name for notebooks/scripts that may still import this.
create_arc_comparison_table = create_comparison_table


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
# Styled LaTeX writers (paper-ready format)
# ---------------------------------------------------------------------------

# Display-name remapping for the styled tables
BENCHMARK_DISPLAY = {
    "GPQA": "GPQA-Diamond",
    "AIME": "AIME",
    "SWE-Bench": r"SWE\mbox{-}V",
    "ARC-AGI": "ARC-AGI",
}

RESTRICTION_DISPLAY = {
    "Pareto Restricted All License": "Pareto Restricted All License",
    "Pareto Restricted Open License": "Pareto Restricted Open Weight",
    "All License (no restriction)": "All License (no restriction)",
    "Open License (no restriction)": "Open Weight (no restriction)",
}

# Order in which benchmarks appear in the styled .tex tables
TEX_BENCHMARK_ORDER = ["GPQA", "AIME", "SWE-Bench"]

# Order of the 4 restriction rows per benchmark (for Table 1)
TEX_RESTRICTION_ORDER = [
    "Pareto Restricted All License",
    "Pareto Restricted Open License",
    "All License (no restriction)",
    "Open License (no restriction)",
]


def _is_missing(val) -> bool:
    """True if a numeric/string field should be rendered as '---'."""
    if val is None:
        return True
    s = str(val).strip().lower()
    if s in ("", "n/a", "nan", "---", "none"):
        return True
    if "nan" in s:
        return True
    return False


def _fmt_factor(val) -> str:
    if _is_missing(val):
        return "---"
    try:
        return f"{float(val):.3f}"
    except (TypeError, ValueError):
        return "---"


def _fmt_ci(ci_str) -> str:
    """Render a CI string '[lo, hi]'. Returns '[---, ---]' if missing/garbage."""
    if _is_missing(ci_str):
        return "[---, ---]"
    s = str(ci_str).strip()
    # Drop pathologically wide CIs (e.g. open/sparse SWE rows reporting up to ~1e5)
    if any(tok in s.lower() for tok in ("nan",)):
        return "[---, ---]"
    return s


def _fmt_n(val) -> str:
    if _is_missing(val):
        return "---"
    try:
        n = int(float(val))
        return str(n) if n > 0 else "---"
    except (TypeError, ValueError):
        return "---"


def _fmt_r2(val, ndigits=4) -> str:
    if _is_missing(val):
        return "---"
    try:
        return f"{float(val):.{ndigits}f}"
    except (TypeError, ValueError):
        return "---"


def _row_prefix(i: int, bench_label: str) -> str:
    """
    Return the LaTeX cell prefix for the i-th restriction row (0..3) of a benchmark.
    Row 0 carries the benchmark name; rows 1 and 3 are gray-shaded.
    """
    if i == 0:
        return bench_label
    if i % 2 == 1:
        return r"\rowcolor{restrgray} \cellcolor{white}"
    return r"\cellcolor{white}"


def write_table1_styled_latex(table_df: pd.DataFrame, path: str, caption: str, label: str):
    """
    Write Table 1 (Annual Reduction Factor) in the user's preferred LaTeX style:
    - Multi-row benchmark grouping with \\cline{2-6}
    - Alternating gray rows via \\rowcolor{restrgray}
    - Renamed restrictions (Open License -> Open Weight) and benchmarks
      (GPQA -> GPQA-Diamond, SWE-Bench -> SWE\\mbox{-}V).
    - Missing/sparse rows rendered as '---'.

    Expects `table_df` columns: Benchmark, Restriction, Year Decrease Factor,
    90% CI, n, R²  (the first column may be blank-filled — we recover the
    benchmark from row position if needed).
    """
    # Re-fill any blank Benchmark cells (the CSV blanks them for visual grouping).
    df = table_df.copy()
    if "Benchmark" in df.columns:
        df["Benchmark"] = df["Benchmark"].replace("", np.nan).ffill()

    # Index by (Benchmark, Restriction) for easy lookup; missing combos -> blanks.
    df_idx = df.set_index(["Benchmark", "Restriction"])

    lines = []
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append("")
    lines.append(r"\begin{tabular}{|l|l|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Benchmark} & \textbf{Restriction} & "
        r"\textbf{Annual Reduction Factor} & \textbf{90\% CI} & "
        r"\textbf{n} & \textbf{$R^2$} \\"
    )
    lines.append(r"\hline")
    lines.append("")

    for bench_key in TEX_BENCHMARK_ORDER:
        if bench_key not in df["Benchmark"].unique():
            continue
        bench_label = BENCHMARK_DISPLAY.get(bench_key, bench_key)
        lines.append(rf"% ===== {bench_label} =====")
        for i, restr in enumerate(TEX_RESTRICTION_ORDER):
            restr_display = RESTRICTION_DISPLAY[restr]
            try:
                row = df_idx.loc[(bench_key, restr)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                factor = _fmt_factor(row.get("Year Decrease Factor"))
                ci = _fmt_ci(row.get("90% CI"))
                n = _fmt_n(row.get("n"))
                r2 = _fmt_r2(row.get("R²"))
            except KeyError:
                factor, ci, n, r2 = "---", "[---, ---]", "---", "---"

            prefix = _row_prefix(i, bench_label)
            line = (
                f"{prefix} & {restr_display} & {factor} & {ci} & {n} & {r2} \\\\"
            )
            if i < len(TEX_RESTRICTION_ORDER) - 1:
                line += r" \cline{2-6}"
            lines.append(line)
        lines.append(r"\hline")
        lines.append("")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Table 2: logit(score) ~ time [+ log(price)] across Pareto / Frontier / All
# ---------------------------------------------------------------------------

def _filter_score_frontier(df: pd.DataFrame, score_col: str, date_col: str = "Release Date") -> pd.DataFrame:
    """
    Keep only the cumulative-best-score rows over time (the SOTA frontier).
    For each release date take the best score, then keep rows where the score
    strictly exceeds the running maximum.
    """
    work = df.dropna(subset=[score_col, date_col]).copy()
    work = work.sort_values([date_col, score_col], ascending=[True, False])
    # One representative point per date (the day's best)
    work = work.groupby(date_col, as_index=False).first()
    keep = []
    prev_max = -np.inf
    for idx, score in zip(work.index, work[score_col].values):
        if score > prev_max:
            keep.append(idx)
            prev_max = score
    return work.loc[keep].copy()


def _filter_pareto_frontier_over_time(
    df: pd.DataFrame, score_col: str, price_col: str
) -> pd.DataFrame:
    """
    Replicate the per-date Pareto frontier filter used by
    `plot_price_mmlu_regression` (cheaper price + higher score).
    """
    df_reg = df.sort_values("Release Date").copy()
    pareto_indices = []
    for date in df_reg["Release Date"].unique():
        avail = df_reg[df_reg["Release Date"] <= date].copy()
        avail = avail.sort_values([price_col, score_col])
        frontier = []
        for i, row in avail.iterrows():
            dominated = False
            for j in frontier:
                fr = avail.loc[j]
                if (
                    fr[score_col] >= row[score_col]
                    and fr[price_col] <= row[price_col]
                    and (fr[score_col] > row[score_col] or fr[price_col] < row[price_col])
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(i)
                new_frontier = []
                for j in frontier[:-1]:
                    fr = avail.loc[j]
                    if not (
                        row[score_col] >= fr[score_col]
                        and row[price_col] <= fr[price_col]
                        and (row[score_col] > fr[score_col] or row[price_col] < fr[price_col])
                    ):
                        new_frontier.append(j)
                frontier = new_frontier + [i]
        for i, row in df_reg[df_reg["Release Date"] == date].iterrows():
            if i in frontier:
                pareto_indices.append(i)
    pareto_indices = list(set(pareto_indices))
    return df_reg.loc[pareto_indices].copy()


def _prep_score_regression_df(
    df: pd.DataFrame, score_col: str, price_col: str
) -> pd.DataFrame:
    """Clean scores & prices, drop NaNs, return a fresh DataFrame ready for filtering."""
    work = df.copy()
    work["Release Date"] = pd.to_datetime(work["Release Date"], errors="coerce")
    work[score_col] = pd.to_numeric(
        work[score_col].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )
    work[price_col] = pd.to_numeric(
        work[price_col].astype(str).str.replace("[$,]", "", regex=True),
        errors="coerce",
    )
    work = work.dropna(subset=["Release Date", score_col, price_col])
    work = work[(work[score_col] > 0) & (work[price_col] > 0)]
    return work


def _logit(p: np.ndarray, eps: float = 0.005) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_score_time_regression(
    df: pd.DataFrame,
    score_col: str,
    price_col: str,
    sample: str,
    with_price_control: bool,
):
    """
    Fit logit(score/100) ~ alpha * date_ordinal + (beta * log(price))? + c
    on the chosen sample.  Time coefficient and SE are reported on a /year
    basis (multiply daily values by 365).
    Returns dict with keys: n, time_coef, time_se, price_coef, r2.
    """
    work = _prep_score_regression_df(df, score_col, price_col)

    if sample == "Pareto":
        work = _filter_pareto_frontier_over_time(work, score_col, price_col)
    elif sample == "Frontier":
        work = _filter_score_frontier(work, score_col)
    # "All" -> no filter

    if len(work) < 3:
        return {"n": len(work), "time_coef": None, "time_se": None,
                "price_coef": None, "r2": None}

    work = work.sort_values("Release Date").copy()
    work["Date_Ordinal"] = work["Release Date"].map(datetime.toordinal)

    y = _logit(work[score_col].values / 100.0)
    if with_price_control:
        X = np.column_stack(
            [work["Date_Ordinal"].values, np.log(work[price_col].values)]
        )
    else:
        X = work["Date_Ordinal"].values.reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)

    n = len(work)
    p_dim = X.shape[1]
    if n - p_dim - 1 > 0:
        residuals = y - y_pred
        mse = float(np.sum(residuals ** 2) / (n - p_dim - 1))
        Xc = X - X.mean(axis=0, keepdims=True)
        try:
            cov = np.linalg.inv(Xc.T @ Xc) * mse
            se_alpha = float(np.sqrt(cov[0, 0]))
        except np.linalg.LinAlgError:
            se_alpha = np.nan
    else:
        se_alpha = np.nan

    time_coef = float(model.coef_[0]) * 365.0
    time_se = se_alpha * 365.0 if not np.isnan(se_alpha) else None
    price_coef = float(model.coef_[1]) if with_price_control else None

    return {
        "n": n,
        "time_coef": time_coef,
        "time_se": time_se,
        "price_coef": price_coef,
        "r2": float(r2),
    }


def compute_table2_data(df_gpqa, df_aime, df_swe):
    """
    Run the score-on-time regressions for every (benchmark x sample x control)
    cell in Table 2.  Returns a list of row dicts in display order.
    """
    benchmarks = [
        ("GPQA",      df_gpqa, "epoch_gpqa",   "Benchmark Cost USD"),
        ("AIME",      df_aime, "oneshot_AIME", "Benchmark Cost USD"),
        ("SWE-Bench", df_swe,  "epoch_swe",    "Benchmark Cost USD"),
    ]
    samples = ["Pareto", "Frontier", "All"]
    controls = [False, True]

    rows = []
    for bench_key, df, score_col, price_col in benchmarks:
        for sample in samples:
            for with_price in controls:
                res = fit_score_time_regression(
                    df, score_col, price_col, sample, with_price
                )
                rows.append({
                    "benchmark": bench_key,
                    "sample": sample,
                    "with_price": with_price,
                    **res,
                })
    return rows


def _fmt_coef(val, ndigits=2, allow_missing=True):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---" if allow_missing else "0"
    fmt = f"{{:.{ndigits}f}}"
    s = fmt.format(val)
    # Wrap negative numbers in math mode (matches user's $-0.021$ style).
    if s.startswith("-"):
        return f"${s}$"
    return s


def write_table2_styled_latex(rows, path, caption, label):
    """Render Table 2 (logit performance trend) in the user's preferred style."""
    sample_labels = {
        "Pareto":   "Pareto",
        "Frontier": "Frontier",
        "All":      "All",
    }

    lines = []
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append("")
    lines.append(r"\begin{tabular}{|l|l|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Benchmark} &" + "\n" +
        r"\textbf{Sample} &" + "\n" +
        r"\textbf{n} &" + "\n" +
        r"\shortstack{\textbf{Time}\\\textbf{Coef}} &" + "\n" +
        r"\shortstack{\textbf{Time}\\\textbf{SE}} &" + "\n" +
        r"\shortstack{\textbf{Price}\\\textbf{Coef}} &" + "\n" +
        r"\textbf{$R^2$} \\"
    )
    lines.append(r"\hline")
    lines.append("")

    # Group rows by benchmark, in our display order
    bench_order = TEX_BENCHMARK_ORDER
    grouped = {b: [] for b in bench_order}
    for r in rows:
        if r["benchmark"] in grouped:
            grouped[r["benchmark"]].append(r)

    for bench_key in bench_order:
        bench_rows = grouped.get(bench_key, [])
        if not bench_rows:
            continue
        bench_label = BENCHMARK_DISPLAY.get(bench_key, bench_key)
        lines.append(rf"% ===== {bench_label} =====")

        # Order: Pareto/Without, Pareto/With, Frontier/Without, Frontier/With,
        # All/Without, All/With  -> 6 rows
        ordered = []
        for sample in ["Pareto", "Frontier", "All"]:
            for with_price in [False, True]:
                match = [
                    r for r in bench_rows
                    if r["sample"] == sample and r["with_price"] == with_price
                ]
                ordered.append(match[0] if match else None)

        for i, r in enumerate(ordered):
            if i == 0:
                prefix = bench_label
            elif i % 2 == 1:
                prefix = r"\rowcolor{restrgray} \cellcolor{white}"
            else:
                prefix = r"\cellcolor{white}"

            if r is None:
                sample_label = "---"
                n_str = "---"
                tcoef = tse = pcoef = r2 = "---"
            else:
                sample_label = sample_labels.get(r["sample"], r["sample"])
                control_str = "With Price Control" if r["with_price"] else "Without Price Control"
                sample_label = f"{sample_label}, {control_str}"
                n_str = _fmt_n(r["n"])
                tcoef = _fmt_coef(r["time_coef"], 2)
                tse = _fmt_coef(r["time_se"], 3)
                pcoef = _fmt_coef(r["price_coef"], 3) if r["with_price"] else "---"
                r2 = _fmt_r2(r["r2"], 2)

            line = (
                f"{prefix} & {sample_label} & {n_str} & "
                f"{tcoef} & {tse} & {pcoef} & {r2} \\\\"
            )
            if i < len(ordered) - 1:
                line += r" \cline{2-7}"
            lines.append(line)

        lines.append(r"\hline")
        lines.append("")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


def write_table2_csv(rows, path):
    """Save Table 2 results as a tidy CSV."""
    df = pd.DataFrame([
        {
            "Benchmark": r["benchmark"],
            "Sample": r["sample"],
            "Price Control": "Yes" if r["with_price"] else "No",
            "n": r["n"],
            "Time Coef (per year)": r["time_coef"],
            "Time SE (per year)": r["time_se"],
            "Price Coef (log price)": r["price_coef"],
            "R^2": r["r2"],
        }
        for r in rows
    ])
    df.to_csv(path, index=False)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load generated benchmark datasets. `generate_new_csv.py` produces these
    # from data/new_econ_eval.csv in Step 1 of generate_all.sh.
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

    print(f"Loaded: GPQA={len(df_gpqa)}, AIME={len(df_aime)}, SWE={len(df_swe)}")

    # ── Raw table (no hardware adjustment) ──
    print("\n" + "=" * 80)
    print("REGRESSION TABLE — RAW (no hardware adjustment)")
    print("=" * 80)

    table_raw = create_comparison_table(
        df_gpqa, df_aime, df_swe,
        hardware_gain_factor=1.0,
    )
    print("\n")
    print(table_raw.to_string(index=False))

    table_raw.to_csv("results_data/regression_comparison_table_raw.csv", index=False)
    table_raw.to_csv("results_data/regression_comparison_table.csv", index=False)
    print("\nSaved results_data/regression_comparison_table_raw.csv")
    print("Saved results_data/regression_comparison_table.csv")
    write_table1_styled_latex(
        table_raw,
        "results_data/regression_comparison_table_raw.tex",
        caption="Regression Results (Raw, No Hardware Adjustment)",
        label="tab:regression_results_raw",
    )

    # ── Hardware-adjusted table (divide by 1/0.7 ≈ 1.43) ──
    print("\n" + "=" * 80)
    print("REGRESSION TABLE — HARDWARE-ADJUSTED (÷ 1/0.7)")
    print("=" * 80)

    table_hw = create_comparison_table(
        df_gpqa, df_aime, df_swe,
        hardware_gain_factor=(1 / 0.7),
    )
    print("\n")
    print(table_hw.to_string(index=False))

    table_hw.to_csv("results_data/regression_comparison_table_hw_adjusted.csv", index=False)
    print("\nSaved results_data/regression_comparison_table_hw_adjusted.csv")
    write_table1_styled_latex(
        table_hw,
        "results_data/regression_comparison_table_hw_adjusted.tex",
        caption="Regression Results (Hardware-Adjusted, $\\div 1/0.7$)",
        label="tab:regression_results_hw_adjusted",
    )

    # Also keep a "default" Table 1 styled .tex for the paper (uses raw values).
    write_table1_styled_latex(
        table_raw,
        "results_data/regression_comparison_table.tex",
        caption="Annual benchmark-cost reduction factors across benchmarks and "
                "sample restrictions.",
        label="tab:regression_results",
    )

    # ── Table 2: logit performance trend with/without price control ──
    print("\n" + "=" * 80)
    print("REGRESSION TABLE 2 — LOGIT PERFORMANCE TREND")
    print("=" * 80)

    table2_rows = compute_table2_data(df_gpqa, df_aime, df_swe)
    write_table2_csv(table2_rows,
                     "results_data/multi_benchmark_logit_regression.csv")
    write_table2_styled_latex(
        table2_rows,
        "results_data/multi_benchmark_logit_regression.tex",
        caption="Logit performance trend for all models as well as Pareto and "
                "Frontier models (as defined in the paper) with and without "
                "price controls.",
        label="tab:multi_benchmark_regression_rounded",
    )
