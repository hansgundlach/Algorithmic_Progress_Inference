# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from datetime import datetime
from scipy import stats


# %%


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
):
    """
    Plot log(Price) = alpha*time + beta*benchmark + c regression

    Parameters:
    - df: DataFrame with the model data
    - open_license_only: If True, only include models with open licenses
    - min_mmlu: Minimum MMLU score to include (default: 40)
    - max_mmlu: Maximum MMLU score to include (default: 70)
    - price_column: Column name for price data (default: 'Output Price\nUSD/1M Tokens')
    - exclude_dominated: If True, exclude models that are Pareto dominated by earlier models
    - benchmark_col: Column name for benchmark data (default: 'MMLU-Pro (Reasoning & Knowledge)')
    - exclude_reasoning: If True, exclude models with Reasoning_TF = True
    - use_huber: If True, use Huber regression instead of ordinary least squares
    - huber_epsilon: Epsilon parameter for HuberRegressor (default: 1.35)
    - huber_max_iter: Maximum iterations for HuberRegressor (default: 100)
    - pareto_frontier_only: If True, only use Pareto frontier models for the regression
    - use_logit: If True, use logit transformation of benchmark scores (log(score/(100-score)))

    Returns fitted model coefficients and annual decrease rates
    """
    # Column names
    mmlu_col = benchmark_col
    price_col = price_column
    license_col = "License"
    reasoning_col = "Reasoning_TF"

    # Work on a copy
    df_work = df.copy()

    # 1) Convert MMLU "XX%" → float
    df_work[mmlu_col] = (
        df_work[mmlu_col].astype(str).str.replace("%", "", regex=False).astype(float)
    )

    # Apply logit transformation if requested
    if use_logit:
        # Convert percentage scores to proportions (0-1) and apply logit
        # logit(p) = log(p / (1-p)) where p is the proportion
        proportions = df_work[mmlu_col] / 100.0
        # Avoid logit of 0 or 1 by clipping to avoid numerical issues
        proportions = np.clip(proportions, 1e-10, 1 - 1e-10)
        df_work[f"{mmlu_col}_logit"] = np.log(proportions / (1 - proportions))
        mmlu_col_transformed = f"{mmlu_col}_logit"
    else:
        mmlu_col_transformed = mmlu_col

    # 2) Convert price "$X,XXX" → float
    df_work[price_col] = (
        df_work[price_col].astype(str).str.replace("[$,]", "", regex=True)
    )
    df_work[price_col] = pd.to_numeric(df_work[price_col], errors="coerce")

    # 3) Optionally filter to open-license only
    if open_license_only:
        df_work = df_work[
            df_work[license_col].notna()
            & df_work[license_col].str.contains("open", case=False, na=False)
        ]

    # 4) Optionally filter out reasoning models
    if exclude_reasoning and reasoning_col in df_work.columns:
        df_work = df_work[df_work[reasoning_col] != True]

    # 5) Filter to rows with all necessary data
    df_sub = df_work.dropna(subset=["Release Date", price_col, mmlu_col])
    df_sub = df_sub[(df_sub[price_col] > 0) & (df_sub[mmlu_col] > 0)]

    # 6) Filter by MMLU range
    df_sub = df_sub[(df_sub[mmlu_col] >= min_mmlu) & (df_sub[mmlu_col] <= max_mmlu)]

    # 7) Optionally filter out Pareto dominated models (this affects data display)
    df_sub_display = df_sub.copy()  # Keep original for display
    if exclude_dominated:
        df_sub_display = df_sub_display.sort_values("Release Date")
        non_dominated = []

        for i, row in df_sub_display.iterrows():
            # Check if this model is dominated by any previous model
            dominated = False
            for j in non_dominated:
                prev_row = df_sub_display.loc[j]
                # A model is dominated if there exists a previous model with:
                # 1. Better or equal MMLU score AND
                # 2. Lower or equal price
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

                # Also remove any previously added models that this one dominates
                new_non_dominated = []
                for j in non_dominated[:-1]:  # All except the one we just added
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

    # 8) For regression, decide which data to use
    if pareto_frontier_only:
        # Identify Pareto frontier models at each point in time
        df_regression = df_sub.sort_values("Release Date").copy()
        pareto_indices = []

        for date in df_regression["Release Date"].unique():
            # Get all models available at this date
            available_models = df_regression[
                df_regression["Release Date"] <= date
            ].copy()

            # Find Pareto frontier at this date
            available_models = available_models.sort_values([price_col, mmlu_col])
            frontier_indices = []

            for i, row in available_models.iterrows():
                # Check if this model is on the Pareto frontier
                dominated = False
                for j in frontier_indices:
                    frontier_row = available_models.loc[j]
                    # A model is dominated if there exists another model with:
                    # 1. Better or equal MMLU score AND
                    # 2. Lower or equal price
                    # AND at least one is strictly better
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
                    # Remove any previously added models that this one dominates
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

            # Add models released exactly on this date that are on the frontier
            current_date_models = df_regression[df_regression["Release Date"] == date]
            for i, row in current_date_models.iterrows():
                if i in frontier_indices:
                    pareto_indices.append(i)

        # Remove duplicates and use for regression
        pareto_indices = list(set(pareto_indices))
        df_regression = df_regression.loc[pareto_indices]
    else:
        df_regression = df_sub.copy()

    if len(df_regression) < 3:
        print(
            f"Warning: Only {len(df_regression)} data points available for regression. Need at least 3."
        )
        return None, None, None

    # 9) Prepare variables for regression
    df_regression = df_regression.sort_values("Release Date").copy()
    df_regression["Date_Ordinal"] = df_regression["Release Date"].map(
        datetime.toordinal
    )

    # Features: time and benchmark (with optional logit transformation)
    X = np.column_stack(
        [
            df_regression["Date_Ordinal"].values,
            df_regression[mmlu_col_transformed].values,
        ]
    )

    # Target: log(Price)
    y = np.log(df_regression[price_col].values)

    # 10) Fit regression (OLS or Huber)
    if use_huber:
        model = HuberRegressor(epsilon=huber_epsilon, max_iter=huber_max_iter).fit(X, y)
        alpha, beta = model.coef_
        c = model.intercept_
        # HuberRegressor does not provide R^2 directly, so we compute it manually
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

    # 11) Calculate annual decrease rates
    # alpha is change in log(price) per day, so annual change is alpha * 365
    annual_log_change = alpha * 365
    annual_pct_change = (np.exp(annual_log_change) - 1) * 100
    factor_change_per_year = np.exp(annual_log_change)

    # Always express as factor decrease to show values < 1 for decreasing trends
    # When alpha < 0: price decreasing, factor_change < 1, so 1/factor_change > 1 (factor decrease)
    # When alpha > 0: price increasing, factor_change > 1, so 1/factor_change < 1 (factor decrease)
    factor_decrease_per_year = 1 / factor_change_per_year

    # 12) Calculate confidence intervals for the time coefficient (only for OLS)
    if not use_huber:
        n = len(df_regression)
        p = 2  # number of predictors (time and GPQA)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - p - 1)

        # Calculate variance-covariance matrix
        X_mean_centered = X - np.mean(X, axis=0)
        cov_matrix = np.linalg.inv(X_mean_centered.T.dot(X_mean_centered)) * mse

        # Standard error for alpha (time coefficient)
        se_alpha = np.sqrt(cov_matrix[0, 0])
        se_annual = se_alpha * 365  # Standard error for annual coefficient

        # t-statistic for 90% confidence interval
        t_stat = stats.t.ppf(0.95, n - p - 1)

        # Confidence interval for annual log change
        annual_log_change_lower = annual_log_change - t_stat * se_annual
        annual_log_change_upper = annual_log_change + t_stat * se_annual

        # Convert to factor change confidence interval
        factor_change_lower = np.exp(annual_log_change_lower)
        factor_change_upper = np.exp(annual_log_change_upper)

        # Always express as factor decrease confidence interval (reciprocal to match factor_decrease_per_year)
        factor_decrease_lower = 1 / factor_change_upper
        factor_decrease_upper = 1 / factor_change_lower
    else:
        # HuberRegressor does not provide standard errors/confidence intervals
        factor_change_lower = None
        factor_change_upper = None
        factor_decrease_lower = None
        factor_decrease_upper = None

    # 13) Generate predictions for plotting
    min_ord, max_ord = (
        df_regression["Date_Ordinal"].min(),
        df_regression["Date_Ordinal"].max(),
    )
    x_range = np.linspace(min_ord, max_ord, 100)
    x_dates = [datetime.fromordinal(int(d)) for d in x_range]

    # For visualization, we'll show the trend at median benchmark value
    median_benchmark = df_regression[mmlu_col].median()
    if use_logit:
        # Convert median percentage to logit
        median_proportion = median_benchmark / 100.0
        median_proportion = np.clip(median_proportion, 1e-10, 1 - 1e-10)
        median_logit = np.log(median_proportion / (1 - median_proportion))
        X_pred = np.column_stack([x_range, np.full(len(x_range), median_logit)])
    else:
        X_pred = np.column_stack([x_range, np.full(len(x_range), median_benchmark)])
    y_pred_plot = model.predict(X_pred)

    # 14) Plot results
    plt.figure(figsize=(12, 8))

    # Color points by MMLU score for better visualization (use display data)
    scatter = plt.scatter(
        df_sub_display["Release Date"],
        df_sub_display[price_col],
        c=df_sub_display[mmlu_col],
        cmap="viridis",
        alpha=0.7,
        s=60,
        label="Data points",
    )

    # If using Pareto frontier for regression, highlight those points
    if pareto_frontier_only:
        plt.scatter(
            df_regression["Release Date"],
            df_regression[price_col],
            facecolors="none",
            edgecolors="red",
            s=100,
            linewidth=2,
            label="Pareto frontier (used for regression)",
        )

    # Add colorbar for MMLU scores
    cbar = plt.colorbar(scatter)
    benchmark_name = benchmark_col.split(" (")[
        0
    ]  # Extract the main part of the benchmark name
    cbar.set_label(f"{benchmark_name} Score (%)")

    # Plot regression line (at median benchmark)
    data_source = "Pareto frontier only" if pareto_frontier_only else "all data"
    transform_label = "logit" if use_logit else "linear"
    regression_label = (
        f"{reg_type} Regression ({data_source}, at median {benchmark_name}={median_benchmark:.1f}%, {transform_label})\n"
        f"Annual change: {annual_pct_change:.2f}%/yr\n"
        f"Factor decrease: {factor_decrease_per_year:.3f}×/yr"
        + (
            f" (90% CI: [{factor_decrease_lower:.3f}, {factor_decrease_upper:.3f}])"
            if factor_decrease_lower is not None
            else ""
        )
        + f"\nR² = {r2:.3f}"
    )

    plt.plot(x_dates, np.exp(y_pred_plot), "r-", lw=3, label=regression_label)

    plt.yscale("log")
    plt.xlabel("Release Date")
    plt.ylabel("Price (USD per 1M tokens)")

    lic_label = "open-license only" if open_license_only else "all licenses"
    mmlu_range = f"{benchmark_name} ∈ [{min_mmlu},{max_mmlu}]%"
    price_type = price_col.replace("\n", " ")
    pareto_label = "non-dominated models only" if exclude_dominated else "all models"
    reasoning_label = (
        "excluding reasoning models"
        if exclude_reasoning
        else "including reasoning models"
    )
    frontier_label = (
        "Pareto frontier regression" if pareto_frontier_only else "standard regression"
    )

    transform_desc = "logit" if use_logit else "linear"
    plt.title(
        f"Price vs Time & {benchmark_name} {reg_type} Regression ({lic_label}, {mmlu_range}, {price_type}, {pareto_label}, {reasoning_label}, {frontier_label}, {transform_desc})\n"
        f"log(Price) = {alpha:.6f}×time + {beta:.3f}×{benchmark_name}({transform_desc}) + {c:.3f}"
    )

    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Print detailed results
    print(f"\nRegression Results ({reg_type}):")
    print(f"Data used: {data_source}")
    transform_desc = "logit" if use_logit else "linear"
    print(
        f"Model: log(Price) = {alpha:.6f}×time + {beta:.3f}×{benchmark_name}({transform_desc}) + {c:.3f}"
    )
    print(f"R² score: {r2:.4f}")
    print(f"\nTime coefficient (alpha): {alpha:.6f}")
    print(f"Annual percentage change: {annual_pct_change:.2f}%/yr")
    print(f"Annual factor decrease: {factor_decrease_per_year:.3f}×/yr")
    if factor_decrease_lower is not None:
        print(
            f"90% CI for factor decrease: [{factor_decrease_lower:.3f}, {factor_decrease_upper:.3f}]"
        )

    print(f"{benchmark_name} coefficient (beta): {beta:.3f} ({transform_desc})")
    print(f"Intercept (c): {c:.3f}")
    print(f"\nData points used for regression: {len(df_regression)}")
    print(f"Data points displayed: {len(df_sub_display)}")

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


# %%

df_gpqa = pd.read_csv("price_reduction_models.csv")
print(df_gpqa.columns)
# convert price to float
# df['Output Price\nUSD/1M Tokens'] = df['Output Price\nUSD/1M Tokens'].str.replace('$', '').astype(float)
# df['Lowest Blended Price AA'] = df['Lowest Blended Price AA'].astype(float)
# df['Blended\nUSD/1M Tokens'] = df['Blended\nUSD/1M Tokens'].str.replace('$', '').astype(float)


# convert release date to datetime where release date is not nan
df_gpqa["Release Date"] = pd.to_datetime(df_gpqa["Release Date"])


# # Create Active Parameters column by choosing Known Active Parameters when available, otherwise use Parameters
df_gpqa["Active Parameters"] = np.where(
    df_gpqa["Known Active Parameters"].notna(),
    df_gpqa["Known Active Parameters"],
    df_gpqa["Parameters"],
)
# %%
# df_swe = pd.read_csv("swe_price_reduction_models_edited.csv")
df_swe = pd.read_csv("swe_price_reduction_models.csv")
print(df_swe.columns)
# convert price to float
# df['Output Price\nUSD/1M Tokens'] = df['Output Price\nUSD/1M Tokens'].str.replace('$', '').astype(float)
# df['Lowest Blended Price AA'] = df['Lowest Blended Price AA'].astype(float)
# df['Blended\nUSD/1M Tokens'] = df['Blended\nUSD/1M Tokens'].str.replace('$', '').astype(float)
# convert release date to datetime where release date is not nan
df_swe["Release Date"] = pd.to_datetime(df_swe["Release Date"])
# # Create Active Parameters column by choosing Known Active Parameters when available, otherwise use Parameters
df_swe["Active Parameters"] = np.where(
    df_swe["Known Active Parameters"].notna(),
    df_swe["Known Active Parameters"],
    df_swe["Parameters"],
)

# %%
df_frontier_math = pd.read_csv("frontier_math_price_reduction_models.csv")

df_frontier_math["Release Date"] = pd.to_datetime(df_frontier_math["Release Date"])
# # Create Active Parameters column by choosing Known Active Parameters when available, otherwise use Parameters
df_frontier_math["Active Parameters"] = np.where(
    df_frontier_math["Known Active Parameters"].notna(),
    df_frontier_math["Known Active Parameters"],
    df_frontier_math["Parameters"],
)
# %%

df_aime = pd.read_csv("aime_price_reduction_models.csv")
# print(df_aime.columns)
# convert release date to datetime where release date is not nan
df_aime["Release Date"] = pd.to_datetime(df_aime["Release Date"])
# # Create Active Parameters column by choosing Known Active Parameters when available, otherwise use Parameters
df_aime["Active Parameters"] = np.where(
    df_aime["Known Active Parameters"].notna(),
    df_aime["Known Active Parameters"],
    df_aime["Parameters"],
)
print(len(df_aime))
# %%
# Index(['Model', 'Creator', 'License', 'Context\nWindow',
#        'Artificial Analysis\nIntelligence Index',
#        'MMLU-Pro (Reasoning & Knowledge)',
#        'GPQA Diamond (Scientific Reasoning)',
#        'Humanity's Last Exam (Reasoning & Knowledge)',
#        'LiveCodeBench (Coding)', 'SciCode (Coding)', 'HumanEval (Coding)',
#        'MATH-500 (Quantitative Reasoning)', 'AIME 2024 (Competition Math)',
#        'Multilingual Index (Artificial Analysis)', 'Chatbot Arena',
#        'Blended\nUSD/1M Tokens', 'Input Price\nUSD/1M Tokens',
#        'Output Price\nUSD/1M Tokens', 'Median\nTokens/s', 'P5\nTokens/s',
#        'P25\nTokens/s', 'P75\nTokens/s', 'P95\nTokens/s',
#        'Median\nFirst Chunk (s)', 'First Answer\nToken (s)',
#        'P5\nFirst Chunk (s)', 'P25\nFirst Chunk (s)', 'P75\nFirst Chunk (s)',
#        'P95\nFirst Chunk (s)', 'Total\nResponse (s)', 'Reasoning\nTime (s)',
#        'Further\nAnalysis', 'Release Date', 'Parameters',
#        'Known Active Parameters', 'Lowest Output Price Found AA',
#        'Lowest Input Price AA', 'Lowest Blended Price AA', 'Latency',
#        'token/s', 'Chinese', 'Notes', 'input_tokens_epoch_gpqa',
#        'outpur_tokens_epoch_gpqa', 'epoch_gpqa', 'price input lowest',
#        'price output lowest', 'total price lowest'],
#       dtype='object')


# Lowest Output Price Found AA
# Lowest Input Price AA

# benchmark1_col="MMLU-Pro (Reasoning & Knowledge)",
# benchmark2_col="GPQA Diamond (Scientific Reasoning)",
# benchmark3_col="LiveCodeBench (Coding)",

# Example usage:
# Assuming df is loaded with your data
# model, data, results = plot_price_mmlu_regression(df, open_license_only=True, price_column="Lowest Output Price Found AA", exclude_dominated=False)
# model, data, results = plot_price_mmlu_regression(df, open_license_only=True, price_column="Lowest Input Price AA", exclude_dominated=False)

# %%
model, data, results = plot_price_mmlu_regression(
    df_gpqa,
    open_license_only=False,
    price_column="Benchmark Cost USD",
    exclude_dominated=False,
    benchmark_col="epoch_gpqa",
    min_mmlu=25,
    max_mmlu=85,
    exclude_reasoning=False,
    use_huber=False,
    pareto_frontier_only=True,
)
# %%
model, data, results = plot_price_mmlu_regression(
    df_gpqa,
    open_license_only=True,
    price_column="Benchmark Cost USD",
    exclude_dominated=False,
    benchmark_col="epoch_gpqa",
    min_mmlu=0,
    max_mmlu=100,
    exclude_reasoning=False,
    use_huber=False,
    pareto_frontier_only=True,
    use_logit=False,
)
# %%

model, data, results = plot_price_mmlu_regression(
    df_swe,
    open_license_only=True,
    price_column="Benchmark Cost USD",
    exclude_dominated=False,
    benchmark_col="epoch_swe",
    min_mmlu=0,
    max_mmlu=100,
    exclude_reasoning=False,
    use_huber=False,
    pareto_frontier_only=False,
    use_logit=True,
)

# %%
# frontier math analysis
model, data, results = plot_price_mmlu_regression(
    df_frontier_math,
    open_license_only=False,
    price_column="Benchmark Cost USD",
    exclude_dominated=False,
    benchmark_col="frontier accuracy",
    min_mmlu=2,
    max_mmlu=100,
    exclude_reasoning=False,
    use_huber=False,
    pareto_frontier_only=True,
)
# %%
print(df_aime.columns)
# print(df_aime['Benchmark Cost USD'])
# %%
# AIME analysis

model, data, results = plot_price_mmlu_regression(
    df_aime,
    open_license_only=False,
    price_column="Benchmark Cost USD",
    exclude_dominated=False,
    benchmark_col="oneshot_AIME",
    min_mmlu=0,
    max_mmlu=100,
    exclude_reasoning=False,
    use_huber=False,
    pareto_frontier_only=True,
    use_logit=True,
)

# %%
# Example usage with logit transformation:
# model, data, results = plot_price_mmlu_regression(
#     df_aime,
#     open_license_only=True,
#     price_column="Benchmark Cost USD",
#     exclude_dominated=False,
#     benchmark_col="oneshot_AIME",
#     min_mmlu=4,
#     max_mmlu=100,
#     exclude_reasoning=False,
#     use_huber=False,
#     pareto_frontier_only=True,
#     use_logit=True,  # Enable logit transformation
# )
# %%
