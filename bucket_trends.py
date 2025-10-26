# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter
from sklearn.linear_model import LinearRegression, QuantileRegressor
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm
import seaborn as sns
import scienceplots
from scipy import stats

# %%
df = pd.read_csv("price_reduction_models.csv")
print(df.columns)
# convert price to float
# df['Output Price\nUSD/1M Tokens'] = df['Output Price\nUSD/1M Tokens'].str.replace('$', '').astype(float)
# df['Lowest Blended Price AA'] = df['Lowest Blended Price AA'].astype(float)
# df['Blended\nUSD/1M Tokens'] = df['Blended\nUSD/1M Tokens'].str.replace('$', '').astype(float)


# convert release date to datetime where release date is not nan
df["Release Date"] = pd.to_datetime(df["Release Date"])


# # Create Active Parameters column by choosing Known Active Parameters when available, otherwise use Parameters
df["Active Parameters"] = np.where(
    df["Known Active Parameters"].notna(),
    df["Known Active Parameters"],
    df["Parameters"],
)


def plot_combined_record_small_trends(
    open_license_only=False,
    price_col="Lowest Blended Price AA",
    show_model_names=False,
    min_date=datetime(2023, 1, 1),
    confidence_interval=True,
    include_chinese=None,
    benchmark_col="MMLU-Pro (Reasoning & Knowledge)",
    mmlu_ranges=[(30, 50), (50, 70), (70, 90)],
    include_reasoning_models=True,
):
    """
    Plot record-small points and their trend lines for specified MMLU ranges
    on a single graph with enhanced styling. Includes 90% confidence intervals.

    Parameters:
      open_license_only: If True, only include models with open licenses
      price_col: Column name for price data (default: 'Lowest Blended Price AA')
      show_model_names: If True, displays model names next to record-small points
      min_date: If provided, only include models released on or after this date (datetime or string)
                Default is January 1, 2024
      confidence_interval: If True, displays 90% confidence intervals for trend lines
      include_chinese: Filter for Chinese models - if True, only include Chinese models;
                      if False, exclude Chinese models; if None, include all models
      benchmark_col: Column name for the benchmark to use (default: 'MMLU-Pro (Reasoning & Knowledge)')
      mmlu_ranges: List of tuples defining MMLU score ranges (default: [(30, 50), (50, 70), (70, 90)])
      include_reasoning_models: If True, include reasoning models; if False, exclude reasoning models (default: True)
    """
    # Set up the styling with standard matplotlib font
    plt.rcParams["font.family"] = "sans-serif"

    # Create figure with specific dimensions
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    # Set background to white
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Use provided MMLU ranges with custom palette
    palette = sns.color_palette("viridis", n_colors=len(mmlu_ranges))
    markers = [
        "o",
        "s",
        "D",
        "^",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
    ]  # Extended list of markers

    # --- column names ---
    mmlu_col = benchmark_col
    license_col = "License"
    chinese_col = "Chinese"
    reasoning_col = "Reasoning_TF"

    # Store all data for setting axis limits
    all_dates = []
    all_prices = []

    for i, (min_mmlu, max_mmlu) in enumerate(mmlu_ranges):
        color = palette[i]
        marker = markers[
            i % len(markers)
        ]  # Cycle through markers if we have more ranges than markers

        # 1) Work on a copy
        df_work = df.copy()

        # 2) Convert MMLU "XX%" → float
        df_work[mmlu_col] = (
            df_work[mmlu_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

        # 3) Convert price "$X,XXX" → float
        df_work[price_col] = (
            df_work[price_col].astype(str).str.replace("[$,]", "", regex=True)
        )
        df_work[price_col] = pd.to_numeric(df_work[price_col], errors="coerce")

        # 4) Optionally filter to open‐license only
        if open_license_only:
            df_work = df_work[
                df_work[license_col].notna()
                & df_work[license_col].str.contains("open", case=False, na=False)
            ]

        # 4b) Filter by date if min_date is provided
        if min_date is not None:
            if isinstance(min_date, str):
                min_date = pd.to_datetime(min_date)
            df_work = df_work[df_work["Release Date"] >= min_date]

        # 4c) Filter by Chinese models if specified
        if include_chinese is not None:
            if include_chinese:
                # Only include Chinese models (where Chinese column is TRUE)
                df_work = df_work[df_work[chinese_col] == True]
            else:
                # Exclude Chinese models (where Chinese column is TRUE)
                df_work = df_work[
                    (df_work[chinese_col] != True) | (df_work[chinese_col].isna())
                ]

        # 4d) Filter by reasoning models if specified
        if not include_reasoning_models:
            # Exclude reasoning models (where Reasoning_TF column is TRUE)
            df_work = df_work[
                (df_work[reasoning_col] != True) | (df_work[reasoning_col].isna())
            ]

        # 5) Filter to MMLU range
        df_sub = df_work[df_work[mmlu_col].between(min_mmlu, max_mmlu)].copy()

        # 6) Drop missing Release Date or price, remove non‐positive prices
        df_sub = df_sub.dropna(subset=["Release Date", price_col])
        df_sub = df_sub[df_sub[price_col] > 0]

        # Skip if no data
        if len(df_sub) == 0:
            continue

        # 7) Sort & compute ordinal date
        df_sub = df_sub.sort_values("Release Date")
        df_sub["Date_Ordinal"] = df_sub["Release Date"].map(datetime.toordinal)

        # 8) "Record‐small" = running minima of price
        df_sub["Is_Record_Small"] = df_sub[price_col].cummin() == df_sub[price_col]
        record_small = df_sub[df_sub["Is_Record_Small"]].copy()

        # Skip if no record small points
        if len(record_small) == 0:
            continue

        # 9) Linear regression on record small points (log scale)
        X_rec = record_small["Date_Ordinal"].values.reshape(-1, 1)
        y_rec_log = np.log10(record_small[price_col].values)
        rec_ols = LinearRegression().fit(X_rec, y_rec_log)

        # Calculate R^2 value
        r_squared = rec_ols.score(X_rec, y_rec_log)
        print(f"MMLU {min_mmlu}-{max_mmlu}% R² = {r_squared:.3f}")

        # 10) Create prediction line
        min_ord, max_ord = (
            record_small["Date_Ordinal"].min(),
            record_small["Date_Ordinal"].max(),
        )
        x_range = np.arange(min_ord, max_ord + 1)
        x_dates = [datetime.fromordinal(int(d)) for d in x_range]
        y_rec_log_pred = rec_ols.predict(x_range.reshape(-1, 1))

        # 11) Calculate annual decrease rate
        annual_pct_rec = ((10 ** rec_ols.coef_[0]) ** 365 - 1) * 100
        annual_factor_rec = 1 / (10 ** rec_ols.coef_[0]) ** 365

        # Calculate 90% confidence intervals for the slope if requested
        # ci_label = ""
        n = len(X_rec)
        if (
            confidence_interval and n > 2
        ):  # Need at least 3 points for confidence interval
            # Calculate residuals and standard error
            y_pred = rec_ols.predict(X_rec)
            residuals = y_rec_log - y_pred
            mse = np.sum(residuals**2) / (n - 2)
            se = np.sqrt(mse / np.sum((X_rec - np.mean(X_rec)) ** 2))

            # t-value for 90% confidence interval (two-tailed)
            t_val = stats.t.ppf(0.95, n - 2)

            # Confidence interval for slope
            ci_lower = rec_ols.coef_[0] - t_val * se
            ci_upper = rec_ols.coef_[0] + t_val * se

            # Convert to annual factors
            annual_factor_lower = (
                1 / (10**ci_upper) ** 365
            )  # Note: Upper CI of negative slope gives lower factor
            annual_factor_upper = (
                1 / (10**ci_lower) ** 365
            )  # Note: Lower CI of negative slope gives upper factor

            ci_label = (
                f" (90% CI: {annual_factor_lower:.1f}x-{annual_factor_upper:.1f}x)"
            )
            # ci_label = ""

        # 12) Plot record small points with enhanced styling
        sns.scatterplot(
            x=record_small["Release Date"],
            y=record_small[price_col],
            color=color,
            s=100,  # Slightly larger points
            marker=marker,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
            ax=ax,
        )

        # 13) Plot trend line with enhanced styling
        ax.plot(
            x_dates,
            10**y_rec_log_pred,
            color=color,
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            label=f"GPQA-D {min_mmlu}-{max_mmlu}% trend: ({annual_factor_rec:.1f}x cheaper/yr){ci_label}",
        )

        # Plot confidence intervals if we have enough data points and confidence_interval is True
        if confidence_interval and n > 2:
            y_lower = ci_lower * x_range + rec_ols.intercept_
            y_upper = ci_upper * x_range + rec_ols.intercept_

            ax.fill_between(x_dates, 10**y_lower, 10**y_upper, color=color, alpha=0.1)

        # 14) Add model names if requested
        if show_model_names:
            for idx, row in record_small.iterrows():
                ax.annotate(
                    row["Model"],
                    (row["Release Date"], row[price_col]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    alpha=0.9,
                )

        # Store data for axis limits
        all_dates.extend(record_small["Release Date"].tolist())
        all_prices.extend(record_small[price_col].tolist())

    # 15) Enhanced formatting
    ax.set_yscale("log")
    ax.set_xlabel("Date", fontsize=18, fontweight="bold")
    ax.set_ylabel("Benchmark Cost GPQA-D (USD)", fontsize=18, fontweight="bold")

    # Format date axis
    date_formatter = DateFormatter("%b %Y")
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)

    # Title with styling
    lic_label = "open-license only" if open_license_only else "all licenses"
    date_filter = (
        f" (since {min_date.strftime('%b %Y')})" if min_date is not None else ""
    )
    chinese_filter = (
        " (Chinese models only)"
        if include_chinese is True
        else " (non-Chinese models)" if include_chinese is False else ""
    )
    reasoning_filter = " (non-reasoning models)" if not include_reasoning_models else ""
    benchmark_name = benchmark_col.split(" ")[
        0
    ]  # Extract first part of benchmark name for title
    # ax.set_title(
    #     f"Lowest Available Price Trends by {benchmark_name} Range ({lic_label}){date_filter}{chinese_filter}{reasoning_filter}",
    #     fontsize=18,
    #     fontweight="bold",
    #     pad=15,
    # )
    ax.set_title(
        f"Lowest Available Price Trends by GPQA-Diamond Range ({lic_label}){date_filter}{chinese_filter}{reasoning_filter}",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )

    # \n{price_col}

    # Grid styling
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

    # Legend styling
    # legend = ax.legend(
    #     loc='upper right',
    #     bbox_to_anchor=(0.5, -0.1),
    #     fontsize=12,
    #     frameon=True,
    #     fancybox=True,
    #     framealpha=0.95,
    #     edgecolor='gray',
    #     borderpad=1,
    #     ncol=2
    # )
    legend = ax.legend(
        loc="upper right",
        fontsize=17,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="gray",
        borderpad=1,
    )

    # Adjust tick parameters
    ax.tick_params(axis="both", which="major", labelsize=17)

    # Add a subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("gray")
        spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.ylim(10**-2, 10**2.5)
    plt.show()


# %%


# Usage examples:
# For all licenses:
# plot_combined_record_small_trends(open_license_only=False, price_col="Lowest Blended Price AA", min_date=datetime(2023, 12, 1), confidence_interval=True, show_model_names=True)

# For open licenses only with model names:
# plot_combined_record_small_trends(open_license_only=True, price_col="Benchmark Cost USD", confidence_interval=True, min_date=datetime(2023, 12, 1), benchmark_col='epoch_gpqa', show_model_names=False, include_reasoning_models=True, mmlu_ranges=[(30, 40), (40, 50), (50, 60), (60, 70)])


# Blended Price (3:1) USD/1M Tokens

plot_combined_record_small_trends(
    open_license_only=False,
    price_col="Benchmark Cost USD",
    confidence_interval=True,
    min_date=datetime(2024, 4, 1),
    benchmark_col="epoch_gpqa",
    show_model_names=False,
    include_reasoning_models=True,
    mmlu_ranges=[(20, 40), (40, 60), (60, 80)],
)


# %%
