# %%


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
from sklearn.linear_model import LinearRegression
from scipy import stats

# %%

warnings.filterwarnings("ignore")


class BenchmarkPriceAnalyzer:
    def __init__(self, csv_file, input_token_col=None, output_token_col=None):
        """
        Initialize the analyzer with data and token column specifications.

        Args:
            csv_file (str): Path to CSV file with model data
            input_token_col (str): Column name for input tokens
            output_token_col (str): Column name for output tokens
        """
        self.df = pd.read_csv(csv_file)
        self.input_token_col = input_token_col or "input_tokens_epoch_gpqa"
        self.output_token_col = output_token_col or "output_tokens_epoch_gpqa"

        # Find all price date columns
        self.price_date_columns = self._find_price_columns()
        print(f"Found {len(self.price_date_columns)} price date pairs")

    def _find_price_columns(self):
        """Find all input/output price columns with dates."""
        price_cols = []
        input_cols = [
            col
            for col in self.df.columns
            if "input price" in col.lower() and "/" in col
        ]
        output_cols = [
            col
            for col in self.df.columns
            if "output price" in col.lower() and "/" in col
        ]

        # Match input and output columns by date
        for input_col in input_cols:
            # Extract date part (everything before 'input price')
            date_part = input_col.split("input price")[0].strip()
            # Find corresponding output column
            output_col = None
            for out_col in output_cols:
                if out_col.startswith(date_part):
                    output_col = out_col
                    break

            if output_col:
                try:
                    # Parse the date
                    date_str = date_part.rstrip(" ")
                    date_obj = datetime.strptime(date_str, "%m/%d/%Y")
                    price_cols.append(
                        {
                            "date": date_obj,
                            "date_str": date_str,
                            "input_col": input_col,
                            "output_col": output_col,
                        }
                    )
                except ValueError:
                    print(f"Could not parse date from: {date_part}")

        # Sort by date
        price_cols.sort(key=lambda x: x["date"])
        return price_cols

    def compute_benchmark_price(
        self, input_tokens, output_tokens, input_price, output_price
    ):
        """
        Compute benchmark price as: input_price * input_tokens + output_price * output_tokens
        Prices are per million tokens.
        """
        if (
            pd.isna(input_tokens)
            or pd.isna(output_tokens)
            or pd.isna(input_price)
            or pd.isna(output_price)
        ):
            return np.nan

        # Convert to numeric, handling any string issues
        try:
            input_tokens = float(input_tokens)
            output_tokens = float(output_tokens)
            input_price = float(input_price)
            output_price = float(output_price)
        except (ValueError, TypeError):
            return np.nan

        # Prices are per million tokens
        total_cost = (input_price * input_tokens / 1_000_000) + (
            output_price * output_tokens / 1_000_000
        )
        return total_cost

    def analyze_model_trends(self):
        """
        Analyze price trends for each model, applying the filtering constraints.

        Returns:
            dict: Model name -> list of (date, price) tuples
        """
        model_trends = {}

        for idx, row in self.df.iterrows():
            model_name = row["Model"]
            input_tokens = row[self.input_token_col]
            output_tokens = row[self.output_token_col]

            # Skip models without token data
            if pd.isna(input_tokens) or pd.isna(output_tokens):
                continue

            # Compute benchmark prices for each date
            prices_over_time = []
            for price_info in self.price_date_columns:
                input_price = row[price_info["input_col"]]
                output_price = row[price_info["output_col"]]

                benchmark_price = self.compute_benchmark_price(
                    input_tokens, output_tokens, input_price, output_price
                )

                if not pd.isna(benchmark_price):
                    prices_over_time.append(
                        {
                            "date": price_info["date"],
                            "price": benchmark_price,
                            "date_str": price_info["date_str"],
                        }
                    )

            # Apply filtering constraints
            filtered_prices = self._apply_filtering_constraints(prices_over_time)

            if len(filtered_prices) > 0:
                model_trends[model_name] = filtered_prices

        return model_trends

    def _apply_filtering_constraints(self, prices_over_time):
        """
        Apply constraints:
        1. Only include multiple points if price differs
        2. Don't include points where price increases from previous time
        """
        if len(prices_over_time) <= 1:
            return prices_over_time

        # Sort by date
        prices_over_time.sort(key=lambda x: x["date"])

        filtered = []
        last_price = None

        for price_data in prices_over_time:
            current_price = price_data["price"]

            # First point is always included
            if last_price is None:
                filtered.append(price_data)
                last_price = current_price
            else:
                # Only include if price is different and not higher than last price
                if current_price != last_price and current_price <= last_price:
                    filtered.append(price_data)
                    last_price = current_price
                # Also include if it's the same price but we only have one point so far
                elif len(filtered) == 1 and current_price == last_price:
                    # Don't add duplicate prices unless it's a significant time gap
                    # For now, skip exact duplicates
                    pass

        # Special case: if we only have one unique price across all times,
        # just keep the first occurrence
        if len(filtered) == 0 and len(prices_over_time) > 0:
            filtered = [prices_over_time[0]]

        return filtered

    def create_visualization(
        self, model_trends, title="Benchmark Price Trends Over Time"
    ):
        """Create a visualization of the price trends."""
        plt.figure(figsize=(15, 10))

        # Colors for different models
        colormap = plt.cm.get_cmap("tab20", len(model_trends))
        colors = colormap(np.arange(len(model_trends)))

        for i, (model_name, price_data) in enumerate(model_trends.items()):
            if len(price_data) == 0:
                continue

            dates = [p["date"] for p in price_data]
            prices = [p["price"] for p in price_data]

            # Only plot if we have data
            if len(dates) > 0:
                plt.plot(
                    dates,
                    prices,
                    "o-",
                    color=colors[i],
                    label=model_name,
                    alpha=0.7,
                    linewidth=2,
                    markersize=6,
                )

        plt.xlabel("Date", fontsize=12)
        plt.ylabel(
            f"Benchmark Cost (USD)\nUsing {self.input_token_col} & {self.output_token_col}",
            fontsize=12,
        )
        plt.title(title, fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")  # Use log scale for prices as they can vary widely

        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # Legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()

        return plt.gcf()

    def save_results(self, model_trends, output_file="benchmark_trends_data.csv"):
        """Save the results to a CSV file."""
        rows = []
        for model_name, price_data in model_trends.items():
            for p in price_data:
                rows.append(
                    {
                        "Model": model_name,
                        "Date": p["date_str"],
                        "Benchmark_Price_USD": p["price"],
                        "Input_Token_Column": self.input_token_col,
                        "Output_Token_Column": self.output_token_col,
                    }
                )

        results_df = pd.DataFrame(rows)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        return results_df

    def print_summary(self, model_trends):
        """Print a summary of the analysis."""
        print(f"\n=== BENCHMARK PRICE TREND ANALYSIS SUMMARY ===")
        print(f"Token columns used: {self.input_token_col}, {self.output_token_col}")
        print(f"Total models analyzed: {len(model_trends)}")
        print(
            f"Date range: {self.price_date_columns[0]['date_str']} to {self.price_date_columns[-1]['date_str']}"
        )

        # Model with most price points
        max_points = max([len(data) for data in model_trends.values()] + [0])
        models_with_max = [
            name for name, data in model_trends.items() if len(data) == max_points
        ]
        print(
            f"Models with most price points ({max_points}): {', '.join(models_with_max[:3])}"
        )

        # Price ranges
        all_prices = []
        for price_data in model_trends.values():
            all_prices.extend([p["price"] for p in price_data])

        if all_prices:
            print(f"Price range: ${min(all_prices):.4f} to ${max(all_prices):.4f}")
            print(f"Median price: ${np.median(all_prices):.4f}")


def analyze_benchmark_trends(
    csv_file="art_analysis_inf_data.csv",
    input_token_col="input_tokens_epoch_gpqa",
    output_token_col="output_tokens_epoch_gpqa",
    output_plot="benchmark_price_trends.png",
    save_data_file="benchmark_trends_data.csv",
    title="AI Model Benchmark Price Trends Over Time",
    show_plot=True,
):
    """
    Analyze benchmark pricing trends for AI models.

    Args:
        csv_file (str): Path to CSV file with model data
        input_token_col (str): Column name for input tokens
        output_token_col (str): Column name for output tokens
        output_plot (str): Output file for the plot
        save_data_file (str): Output file for the data CSV
        title (str): Title for the plot
        show_plot (bool): Whether to display the plot

    Returns:
        tuple: (model_trends dict, results_df, analyzer object)
    """
    # Create analyzer
    analyzer = BenchmarkPriceAnalyzer(csv_file, input_token_col, output_token_col)

    # Analyze trends
    print("Analyzing price trends...")
    model_trends = analyzer.analyze_model_trends()

    # Create visualization
    print("Creating visualization...")
    fig = analyzer.create_visualization(model_trends, title)
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_plot}")

    # Save results
    results_df = analyzer.save_results(model_trends, save_data_file)

    # Print summary
    analyzer.print_summary(model_trends)

    if show_plot:
        plt.show()

    return model_trends, results_df, analyzer


def main():
    """Default main function for running the script directly."""
    return analyze_benchmark_trends()


if __name__ == "__main__":
    main()

# %%


def run_regression_with_multipoint_data(
    csv_file="art_analysis_inf_data.csv",
    input_token_col="input_tokens_epoch_gpqa",
    output_token_col="output_tokens_epoch_gpqa",
    benchmark_col="epoch_gpqa",
    baseline_date=datetime(2023, 1, 1),
    min_date=None,
    open_license_only=False,
    title_suffix="",
):
    """
    Run log(Price) = alpha*time + beta*log(benchmark) + c using multi-point price observations per model.

    Notes:
    - Time variable uses observation date (provider price date), not model release date.
    - Points are all valid (post-filter) price observations across models.
    """
    analyzer = BenchmarkPriceAnalyzer(csv_file, input_token_col, output_token_col)

    # Build long-form points from analyzer
    model_trends = analyzer.analyze_model_trends()
    point_rows = []
    for model_name, price_list in model_trends.items():
        for p in price_list:
            point_rows.append(
                {
                    "Model": model_name,
                    "ObservationDate": p["date"],
                    "BenchmarkPriceUSD": p["price"],
                }
            )

    if len(point_rows) == 0:
        print("No price observation points available after filtering.")
        return None, None, None

    df_points = pd.DataFrame(point_rows)

    # Join benchmark and metadata from the original df
    meta_cols = [
        col
        for col in ["Model", "Release Date", benchmark_col, "License"]
        if col in analyzer.df.columns
    ]
    if benchmark_col not in meta_cols:
        print(f"Missing benchmark column '{benchmark_col}' in source data.")
        return None, None, None

    df_meta = analyzer.df[meta_cols].copy()
    df_joined = df_points.merge(df_meta, on="Model", how="left")

    # Parse dates
    if df_joined["ObservationDate"].dtype == "object":
        df_joined["ObservationDate"] = pd.to_datetime(
            df_joined["ObservationDate"], errors="coerce"
        )
    if (
        "Release Date" in df_joined.columns
        and df_joined["Release Date"].dtype == "object"
    ):
        df_joined["Release Date"] = pd.to_datetime(
            df_joined["Release Date"], errors="coerce"
        )

    # Optional filters
    if open_license_only and "License" in df_joined.columns:
        df_joined = df_joined[
            df_joined["License"].astype(str).str.contains("open", case=False, na=False)
        ]

    if min_date is not None:
        try:
            min_date_dt = pd.to_datetime(min_date)
            df_joined = df_joined[df_joined["ObservationDate"] >= min_date_dt]
            print(
                f"Applied observation-date filter >= {min_date_dt.date()}: {len(df_joined)} rows remain"
            )
        except Exception:
            print(
                f"Warning: Could not parse min_date '{min_date}'. Skipping time filter."
            )

    # Clean numeric fields
    df_joined[benchmark_col] = pd.to_numeric(df_joined[benchmark_col], errors="coerce")
    df_joined["BenchmarkPriceUSD"] = pd.to_numeric(
        df_joined["BenchmarkPriceUSD"], errors="coerce"
    )

    # Keep valid rows
    df_sub = df_joined.dropna(
        subset=["ObservationDate", "BenchmarkPriceUSD", benchmark_col]
    ).copy()
    df_sub = df_sub[(df_sub["BenchmarkPriceUSD"] > 0) & (df_sub[benchmark_col] > 0)]

    if len(df_sub) < 3:
        print(
            f"Warning: Only {len(df_sub)} data points available after cleaning; need at least 3."
        )
        return None, None, None

    # Time as days since baseline using observation date
    df_sub = df_sub.sort_values("ObservationDate").copy()
    df_sub["Days_Since_Baseline"] = (df_sub["ObservationDate"] - baseline_date).dt.days
    print(
        "Time range (days):",
        df_sub["Days_Since_Baseline"].min(),
        "to",
        df_sub["Days_Since_Baseline"].max(),
    )

    # Regression
    X = np.column_stack(
        [df_sub["Days_Since_Baseline"].values, np.log(df_sub[benchmark_col].values)]
    )
    y = np.log(df_sub["BenchmarkPriceUSD"].values)

    model = LinearRegression().fit(X, y)
    alpha, beta = model.coef_
    c = model.intercept_
    r2 = model.score(X, y)

    n = len(df_sub)
    p = 2
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Annualized stats
    annual_log_change = alpha * 365
    annual_pct_change = (np.exp(annual_log_change) - 1) * 100
    factor_change_per_year = np.exp(annual_log_change)
    factor_decrease_per_year = (
        1 / factor_change_per_year if factor_change_per_year < 1 else None
    )

    # Confidence interval for alpha
    y_pred_fit = model.predict(X)
    residuals = y - y_pred_fit
    mse = np.sum(residuals**2) / (n - p - 1)
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.linalg.inv(X_centered.T.dot(X_centered)) * mse
    se_alpha_daily = np.sqrt(cov_matrix[0, 0])
    se_annual = se_alpha_daily * 365
    t_stat = stats.t.ppf(0.95, n - p - 1)
    annual_log_change_lower = annual_log_change - t_stat * se_annual
    annual_log_change_upper = annual_log_change + t_stat * se_annual
    factor_change_lower = np.exp(annual_log_change_lower)
    factor_change_upper = np.exp(annual_log_change_upper)
    if factor_change_per_year < 1:
        factor_decrease_lower = 1 / factor_change_upper
        factor_decrease_upper = 1 / factor_change_lower
    else:
        factor_decrease_lower = None
        factor_decrease_upper = None

    # Plot scatter and regression line at median benchmark
    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(
        df_sub["ObservationDate"],
        df_sub["BenchmarkPriceUSD"],
        c=df_sub[benchmark_col],
        cmap="viridis",
        alpha=0.7,
        s=60,
        label="Data points",
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"{benchmark_col}")

    min_days, max_days = (
        df_sub["Days_Since_Baseline"].min(),
        df_sub["Days_Since_Baseline"].max(),
    )
    x_range = np.linspace(min_days, max_days, 100)
    x_dates = [baseline_date + pd.Timedelta(days=int(d)) for d in x_range]
    median_benchmark = df_sub[benchmark_col].median()
    X_pred = np.column_stack([x_range, np.full(len(x_range), np.log(median_benchmark))])
    y_pred = model.predict(X_pred)
    plt.plot(
        x_dates,
        np.exp(y_pred),
        "r-",
        lw=3,
        label=(
            f"Regression (at median {benchmark_col}={median_benchmark:.1f})\n"
            f"Annual change: {annual_pct_change:.2f}%/yr\n"
            + (
                f"Factor decrease: {factor_decrease_per_year:.3f}×/yr (90% CI: [{factor_decrease_lower:.3f}, {factor_decrease_upper:.3f}])\n"
                if factor_decrease_per_year is not None
                else f"Factor change: {factor_change_per_year:.3f}×/yr (90% CI: [{factor_change_lower:.3f}, {factor_change_upper:.3f}])\n"
            )
            + f"R² = {r2:.3f}, Adj. R² = {adjusted_r2:.3f}"
        ),
    )

    plt.yscale("log")
    plt.xlabel("Observation Date")
    plt.ylabel("Benchmark Price (USD)")
    time_label = f", from {min_date}" if min_date is not None else ""
    plt.title(
        f"Price vs Observation Time & log({benchmark_col}) Regression{time_label}{(' - ' + title_suffix) if title_suffix else ''}\n"
        f"log(Price) = {alpha:.6f}×days + {beta:.3f}×log({benchmark_col}) + {c:.3f}"
    )
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()

    # Print results
    print("\nRegression Results (multi-point observations):")
    print(
        f"Model: log(Price) = {alpha:.6f}×days + {beta:.3f}×log({benchmark_col}) + {c:.3f}"
    )
    print(f"R²: {r2:.4f}; Adjusted R²: {adjusted_r2:.4f}")
    print(f"Alpha (per day): {alpha:.6f}; Annual % change: {annual_pct_change:.2f}%/yr")
    if factor_decrease_per_year is not None:
        print(
            f"Annual factor decrease: {factor_decrease_per_year:.3f}×/yr; 90% CI: [{factor_decrease_lower:.3f}, {factor_decrease_upper:.3f}]"
        )
    else:
        print(
            f"Annual factor change: {factor_change_per_year:.3f}×/yr; 90% CI: [{factor_change_lower:.3f}, {factor_change_upper:.3f}]"
        )
    print(f"Beta (log {benchmark_col}): {beta:.3f}; Intercept: {c:.3f}")
    print(f"Data points used: {len(df_sub)}")

    results = {
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
        "adjusted_r2_score": adjusted_r2,
    }

    return model, df_sub, results


# %%
# Example usage - run both analyses
if __name__ == "__main__":
    # First, run the basic trend analysis with individual model trajectories
    print("=" * 60)
    print("RUNNING BASIC TREND ANALYSIS")
    print("=" * 60)
    model_trends, results_df, analyzer = analyze_benchmark_trends(
        csv_file="art_analysis_inf_data.csv",
        input_token_col="input_tokens_epoch_gpqa",
        output_token_col="output_tokens_epoch_gpqa",
        output_plot="benchmark_price_trends.png",
        save_data_file="benchmark_trends_data.csv",
        title="AI Model Benchmark Price Trends Over Time",
        show_plot=True,
    )

    # Then run the regression analysis using the multi-point data
    print("\n" + "=" * 60)
    print("RUNNING REGRESSION ANALYSIS")
    print("=" * 60)
    model, df_sub, results = run_regression_with_multipoint_data(
        csv_file="art_analysis_inf_data.csv",
        input_token_col="input_tokens_epoch_gpqa",
        output_token_col="output_tokens_epoch_gpqa",
        benchmark_col="epoch_gpqa",
        baseline_date=datetime(2023, 1, 1),
        min_date="2024-10-01",  # Filter to recent data
        open_license_only=False,
        title_suffix="All models from Oct 2024+",
    )

    plt.show()  # Show all plots

# %%
