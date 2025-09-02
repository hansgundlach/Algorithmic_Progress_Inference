#!/usr/bin/env python3
"""
Script to analyze benchmark pricing trends over time for AI models.

This script computes the cost to run benchmarks for different models over time
and creates visualizations showing price trends, with constraints:
1. Only include multiple points for a model if the benchmark price differs
2. If a model's benchmark price increases from a previous time, exclude that point
"""
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
        colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, len(model_trends)))

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

    def plot_benchmark_price_regression(
        self,
        benchmark_col=None,
        open_license_only=False,
        min_benchmark=None,
        max_benchmark=None,
        exclude_dominated=False,
        min_date=None,
    ):
        """
        Plot log(Price) = alpha*time + beta*log(benchmark) + c regression
        Similar to the notebook function but using benchmark price data.

        Parameters:
        - benchmark_col: Name of benchmark column to use (optional, will use a default)
        - open_license_only: If True, only include models with open licenses (not implemented yet)
        - min_benchmark: Minimum benchmark score to include (optional)
        - max_benchmark: Maximum benchmark score to include (optional)
        - exclude_dominated: If True, exclude models that are Pareto dominated by earlier models
        - min_date: Only include models released on or after this date

        Returns fitted model coefficients and annual decrease rates

        Note: The time variable in the regression is measured in DAYS since baseline (2023-01-01).
              Alpha coefficient represents the daily log change in price.
              Annual changes are calculated by multiplying alpha by 365.
        """
        # Get model trends data
        model_trends = self.analyze_model_trends()

        # Convert model trends to a flat DataFrame and add benchmark data
        rows = []
        for model_name, price_data in model_trends.items():
            # Get the model row from original dataframe
            model_row = self.df[self.df["Model"] == model_name].iloc[0]

            for p in price_data:
                rows.append(
                    {
                        "Model": model_name,
                        "Date": p["date"],
                        "Price": p["price"],
                        "Date_str": p["date_str"],
                        # Add benchmark scores - try different column names if benchmark_col not specified
                        "benchmark_score": self._get_benchmark_score(
                            model_row, benchmark_col
                        ),
                    }
                )

        if len(rows) == 0:
            print("No data available for regression analysis")
            return None, None, None

        df_work = pd.DataFrame(rows)
        print("Available columns:", df_work.columns.tolist())

        # Check if required columns exist
        required_cols = ["Date", "Price", "benchmark_score"]
        missing_cols = [col for col in required_cols if col not in df_work.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None, None, None

        # Debug: Check data in benchmark column
        print(f"\nDebugging benchmark_score column:")
        print(f"Column type: {df_work['benchmark_score'].dtype}")
        print(f"Sample values: {df_work['benchmark_score'].head(10).tolist()}")
        print(f"Non-null count: {df_work['benchmark_score'].notna().sum()}")
        print(
            f"Unique values (first 10): {df_work['benchmark_score'].dropna().unique()[:10]}"
        )

        # Clean and convert benchmark to numeric
        if df_work["benchmark_score"].dtype == "object":
            df_work["benchmark_score"] = (
                df_work["benchmark_score"]
                .astype(str)
                .str.replace("%", "")
                .str.replace(" ", "")
            )
            df_work["benchmark_score"] = df_work["benchmark_score"].replace(
                ["", "nan", "None"], np.nan
            )

        df_work["benchmark_score"] = pd.to_numeric(
            df_work["benchmark_score"], errors="coerce"
        )

        # Apply time filter if specified
        if min_date is not None:
            if isinstance(min_date, str):
                try:
                    min_date_dt = pd.to_datetime(min_date, format="%Y-%m-%d")
                except:
                    try:
                        min_date_dt = pd.to_datetime(min_date)
                    except:
                        print(
                            f"Warning: Could not parse min_date '{min_date}'. Expected format: 'YYYY-MM-DD' or readable date"
                        )
                        min_date_dt = None
            else:
                min_date_dt = min_date

            if min_date_dt is not None:
                initial_count = len(df_work)
                df_work = df_work[df_work["Date"] >= min_date_dt]
                print(
                    f"Applied time filter (>= {min_date_dt.strftime('%Y-%m-%d')}): {initial_count} -> {len(df_work)} rows"
                )

        # Debug: Check data after conversion
        print(f"\nAfter conversion:")
        print(
            f"Valid benchmark_score values: {df_work['benchmark_score'].notna().sum()}"
        )
        print(f"Valid Price values: {df_work['Price'].notna().sum()}")
        print(f"Valid Date values: {df_work['Date'].notna().sum()}")
        print(
            f"Values > 0 in benchmark_score: {(df_work['benchmark_score'] > 0).sum()}"
        )
        print(f"Values > 0 in Price: {(df_work['Price'] > 0).sum()}")

        # Apply benchmark range filters if specified
        if min_benchmark is not None:
            df_work = df_work[df_work["benchmark_score"] >= min_benchmark]
            print(
                f"After filtering for benchmark_score >= {min_benchmark}: {len(df_work)} rows"
            )

        if max_benchmark is not None:
            df_work = df_work[df_work["benchmark_score"] <= max_benchmark]
            print(
                f"After filtering for benchmark_score <= {max_benchmark}: {len(df_work)} rows"
            )

        # Filter to rows with all necessary data
        df_sub = df_work.dropna(subset=required_cols)
        print(f"After dropping NaN values: {len(df_sub)} rows")

        # Filter for positive values
        df_sub = df_sub[(df_sub["Price"] > 0) & (df_sub["benchmark_score"] > 0)]
        print(f"After filtering for positive values: {len(df_sub)} rows")

        # Additional debugging: show what data we have
        if len(df_sub) > 0:
            print(f"\nSample of filtered data:")
            print(df_sub[["Model", "Date", "Price", "benchmark_score"]].head())
        else:
            print(f"\nNo valid data found. Checking individual conditions:")
            print(f"Rows with valid Date: {df_work['Date'].notna().sum()}")
            print(f"Rows with valid Price: {df_work['Price'].notna().sum()}")
            print(
                f"Rows with valid benchmark_score: {df_work['benchmark_score'].notna().sum()}"
            )
            print(f"Rows with Price > 0: {(df_work['Price'] > 0).sum()}")
            print(
                f"Rows with benchmark_score > 0: {(df_work['benchmark_score'] > 0).sum()}"
            )

            # Show some sample data for debugging
            sample_data = df_work[["Model", "Date", "Price", "benchmark_score"]].head(
                10
            )
            print(f"\nSample data for debugging:")
            print(sample_data)

        # Optionally filter out Pareto dominated models
        if exclude_dominated and len(df_sub) > 0:
            df_sub = df_sub.sort_values("Date")
            non_dominated = []

            for i, row in df_sub.iterrows():
                # Check if this model is dominated by any previous model
                dominated = False
                for j in non_dominated:
                    prev_row = df_sub.loc[j]
                    # A model is dominated if there exists a previous model with:
                    # 1. Better or equal benchmark score AND
                    # 2. Lower or equal price
                    if (
                        prev_row["benchmark_score"] >= row["benchmark_score"]
                        and prev_row["Price"] <= row["Price"]
                        and (
                            prev_row["benchmark_score"] > row["benchmark_score"]
                            or prev_row["Price"] < row["Price"]
                        )
                    ):
                        dominated = True
                        break

                if not dominated:
                    non_dominated.append(i)

                    # Also remove any previously added models that this one dominates
                    new_non_dominated = []
                    for j in non_dominated[:-1]:  # All except the one we just added
                        prev_row = df_sub.loc[j]
                        if not (
                            row["benchmark_score"] >= prev_row["benchmark_score"]
                            and row["Price"] <= prev_row["Price"]
                            and (
                                row["benchmark_score"] > prev_row["benchmark_score"]
                                or row["Price"] < prev_row["Price"]
                            )
                        ):
                            new_non_dominated.append(j)

                    non_dominated = new_non_dominated + [i]

            df_sub = df_sub.loc[non_dominated]
            print(f"After Pareto filtering: {len(df_sub)} rows")

        if len(df_sub) < 3:  # Need at least 3 points for 2 predictors + intercept
            print(
                f"Warning: Only {len(df_sub)} data points available. Need at least 3 for regression with time and benchmark."
            )
            return None, None, None

        # Prepare variables for regression
        df_sub = df_sub.sort_values("Date").copy()

        # Use a meaningful baseline date instead of ordinal
        baseline_date = datetime(2023, 1, 1)  # baseline date for time calculation
        df_sub["Days_Since_Baseline"] = (df_sub["Date"] - baseline_date).dt.days

        print(f"\nTime variable details:")
        print(f"Baseline date: {baseline_date.strftime('%Y-%m-%d')}")
        print(
            f"Time range in dataset: {df_sub['Days_Since_Baseline'].min():.0f} to {df_sub['Days_Since_Baseline'].max():.0f} days"
        )
        print(
            f"Time range in years: {df_sub['Days_Since_Baseline'].min()/365:.2f} to {df_sub['Days_Since_Baseline'].max()/365:.2f} years"
        )

        # Features: time (as days since baseline), log(benchmark)
        X = np.column_stack(
            [
                df_sub["Days_Since_Baseline"].values,  # Time in DAYS since baseline
                np.log(df_sub["benchmark_score"].values),
            ]
        )

        # Target: log(total price)
        y = np.log(df_sub["Price"].values)

        # Fit linear regression
        model = LinearRegression().fit(X, y)
        alpha, beta = model.coef_
        c = model.intercept_
        r2 = model.score(X, y)

        # Calculate adjusted R²
        n = len(df_sub)
        p = 2  # number of predictors
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # Calculate annual decrease rates
        # alpha is the daily log change, so multiply by 365 to get annual log change
        annual_log_change = alpha * 365
        annual_pct_change = (np.exp(annual_log_change) - 1) * 100
        factor_change_per_year = np.exp(annual_log_change)

        # Express as factor decrease if price is decreasing
        if factor_change_per_year < 1:
            factor_decrease_per_year = 1 / factor_change_per_year
        else:
            factor_decrease_per_year = None

        # Calculate confidence intervals for the time coefficient
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - p - 1)

        # Calculate variance-covariance matrix
        X_mean_centered = X - np.mean(X, axis=0)
        cov_matrix = np.linalg.inv(X_mean_centered.T.dot(X_mean_centered)) * mse

        # Standard error for alpha (time coefficient)
        se_alpha = np.sqrt(cov_matrix[0, 0])
        se_annual = se_alpha * 365  # Convert daily SE to annual SE

        # t-statistic for 90% confidence interval
        t_stat = stats.t.ppf(0.95, n - p - 1)

        # Confidence interval for annual log change
        annual_log_change_lower = annual_log_change - t_stat * se_annual
        annual_log_change_upper = annual_log_change + t_stat * se_annual

        # Convert to factor change confidence interval
        factor_change_lower = np.exp(annual_log_change_lower)
        factor_change_upper = np.exp(annual_log_change_upper)

        # Express as factor decrease for confidence interval if price is decreasing
        if factor_change_per_year < 1:
            factor_decrease_lower = 1 / factor_change_upper
            factor_decrease_upper = 1 / factor_change_lower
        else:
            factor_decrease_lower = None
            factor_decrease_upper = None

        # Generate predictions for plotting
        min_days, max_days = (
            df_sub["Days_Since_Baseline"].min(),
            df_sub["Days_Since_Baseline"].max(),
        )
        x_range = np.linspace(min_days, max_days, 100)
        x_dates = [baseline_date + pd.Timedelta(days=int(d)) for d in x_range]

        # For visualization, show trend at median values
        median_benchmark = df_sub["benchmark_score"].median()

        X_pred = np.column_stack(
            [x_range, np.full(len(x_range), np.log(median_benchmark))]
        )
        y_pred = model.predict(X_pred)

        # Plot results
        plt.figure(figsize=(14, 8))

        # Color points by benchmark score
        scatter = plt.scatter(
            df_sub["Date"],
            df_sub["Price"],
            c=df_sub["benchmark_score"],
            cmap="viridis",
            alpha=0.7,
            s=60,
            label="Data points",
        )

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Benchmark Score")

        # Plot regression line (at median values)
        if factor_decrease_per_year:
            regression_label = (
                f"Regression (at median benchmark={median_benchmark:.1f})\n"
                f"Annual change: {annual_pct_change:.2f}%/yr\n"
                f"Factor decrease: {factor_decrease_per_year:.3f}×/yr (90% CI: [{factor_decrease_lower:.3f}, {factor_decrease_upper:.3f}])\n"
                f"R² = {r2:.3f}, Adj. R² = {adjusted_r2:.3f}"
            )
        else:
            regression_label = (
                f"Regression (at median benchmark={median_benchmark:.1f})\n"
                f"Annual change: {annual_pct_change:.2f}%/yr\n"
                f"Factor change: {factor_change_per_year:.3f}×/yr (90% CI: [{factor_change_lower:.3f}, {factor_change_upper:.3f}])\n"
                f"R² = {r2:.3f}, Adj. R² = {adjusted_r2:.3f}"
            )

        plt.plot(x_dates, np.exp(y_pred), "r-", lw=3, label=regression_label)

        plt.yscale("log")
        plt.xlabel("Date")
        plt.ylabel(
            f"Benchmark Cost (USD)\nUsing {self.input_token_col} & {self.output_token_col}"
        )

        pareto_label = (
            "non-dominated models only" if exclude_dominated else "all models"
        )
        time_label = f", from {min_date}" if min_date is not None else ""

        plt.title(
            f"Benchmark Price vs Time & log(Benchmark) Regression ({pareto_label}{time_label})\n"
            f"log(Price) = {alpha:.6f}×days + {beta:.3f}×log(benchmark) + {c:.3f}"
        )

        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

        # Print detailed results
        print(f"\nRegression Results:")
        print(
            f"Model: log(Price) = {alpha:.6f}×days + {beta:.3f}×log(benchmark) + {c:.3f}"
        )
        print(f"R² score: {r2:.4f}")
        print(f"Adjusted R² score: {adjusted_r2:.4f}")
        print(f"\nTime coefficient (alpha): {alpha:.6f} per day")
        print(f"This represents a daily log change of {alpha:.6f}")
        print(f"Annual percentage change: {annual_pct_change:.2f}%/yr")

        if factor_decrease_per_year:
            print(f"Annual factor decrease: {factor_decrease_per_year:.3f}×/yr")
            print(
                f"90% CI for factor decrease: [{factor_decrease_lower:.3f}, {factor_decrease_upper:.3f}]"
            )
        else:
            print(f"Annual factor change: {factor_change_per_year:.3f}×/yr")
            print(
                f"90% CI for factor change: [{factor_change_lower:.3f}, {factor_change_upper:.3f}]"
            )

        print(f"log(benchmark) coefficient (beta): {beta:.3f}")
        print(f"Intercept (c): {c:.3f}")
        print(f"\nData points used: {len(df_sub)}")

        return (
            model,
            df_sub,
            {
                "alpha": alpha,
                "beta": beta,
                "c": c,
                "annual_pct_change": annual_pct_change,
                "factor_change_per_year": factor_change_per_year,
                "factor_decrease_per_year": (
                    factor_decrease_per_year if factor_change_per_year < 1 else None
                ),
                "factor_change_ci_lower": factor_change_lower,
                "factor_change_ci_upper": factor_change_upper,
                "factor_decrease_ci_lower": (
                    factor_decrease_lower if factor_change_per_year < 1 else None
                ),
                "factor_decrease_ci_upper": (
                    factor_decrease_upper if factor_change_per_year < 1 else None
                ),
                "r2_score": r2,
                "adjusted_r2_score": adjusted_r2,
            },
        )

    def _get_benchmark_score(self, model_row, benchmark_col=None):
        """Get benchmark score for a model, trying different column names if benchmark_col not specified."""
        if benchmark_col and benchmark_col in model_row.index:
            return model_row[benchmark_col]

        # Try common benchmark column names
        possible_cols = ["epoch_gpqa", "GPQA", "mmlu", "MMLU", "benchmark_score"]
        for col in possible_cols:
            if col in model_row.index and pd.notna(model_row[col]):
                return model_row[col]

        # If no benchmark found, return NaN
        return np.nan


def analyze_benchmark_trends(
    csv_file="art_analysis_inf_data.csv",
    input_token_col="input_tokens_epoch_gpqa",
    output_token_col="output_tokens_epoch_gpqa",
    output_plot="benchmark_price_trends.png",
    save_data_file="benchmark_trends_data.csv",
    title="AI Model Benchmark Price Trends Over Time",
    show_plot=True,
    include_regression=False,
    regression_min_date=None,
    exclude_outliers=False,
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
        include_regression (bool): Whether to include regression analysis
        regression_min_date (str): Minimum date for regression analysis (YYYY-MM-DD format)
        exclude_outliers (bool): Whether to exclude outliers in regression

    Returns:
        tuple: (model_trends dict, results_df, analyzer object, regression_results)
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

    # Optionally perform regression analysis
    regression_results = None
    if include_regression:
        print("\nPerforming regression analysis...")
        regression_model, regression_df, regression_results = (
            analyzer.create_regression_analysis(
                model_trends,
                title=f"Regression Analysis: {title}",
                min_date=regression_min_date,
                exclude_outliers=exclude_outliers,
            )
        )

    if show_plot:
        plt.show()

    return model_trends, results_df, analyzer, regression_results


def run_regression_analysis(
    csv_file="art_analysis_inf_data.csv",
    input_token_col="input_tokens_epoch_gpqa",
    output_token_col="output_tokens_epoch_gpqa",
    min_date=None,
    exclude_outliers=False,
    title="Benchmark Price Regression Analysis",
):
    """
    Run only the regression analysis on benchmark price trends.

    Args:
        csv_file (str): Path to CSV file with model data
        input_token_col (str): Column name for input tokens
        output_token_col (str): Column name for output tokens
        min_date (str): Minimum date for analysis (YYYY-MM-DD format)
        exclude_outliers (bool): Whether to exclude statistical outliers
        title (str): Title for the regression plot

    Returns:
        tuple: (regression_model, data_df, regression_results)
    """
    # Create analyzer and get trends
    analyzer = BenchmarkPriceAnalyzer(csv_file, input_token_col, output_token_col)
    model_trends = analyzer.analyze_model_trends()

    # Run regression analysis
    return analyzer.create_regression_analysis(
        model_trends, title=title, min_date=min_date, exclude_outliers=exclude_outliers
    )


def main():
    """Default main function for running the script directly."""
    return analyze_benchmark_trends()


if __name__ == "__main__":
    main()

# %%
# Example usage for regression analysis:

# Basic trend analysis (without regression)
model_trends, results_df, analyzer, _ = analyze_benchmark_trends()

# NEW: Use the enhanced regression analysis with time + benchmark variables
# This matches the notebook function closely
analyzer = BenchmarkPriceAnalyzer("art_analysis_inf_data.csv")
model, data, results = analyzer.plot_benchmark_price_regression(
    benchmark_col="epoch_gpqa",  # Specify benchmark column
    min_benchmark=0,
    max_benchmark=100,
    exclude_dominated=False,
    min_date="2024-10-01",  # Only include models from October 2024 onwards
)

if results:
    print(f"Annual price change: {results['annual_pct_change']:.2f}%/yr")
    if results["factor_decrease_per_year"]:
        print(f"Factor decrease per year: {results['factor_decrease_per_year']:.3f}×")
    print(f"Benchmark coefficient (beta): {results['beta']:.3f}")

# Or the simpler regression (time only, previous version)
regression_model, regression_df, regression_results = run_regression_analysis(
    min_date="2024-01-01",
    exclude_outliers=True,
    title="Benchmark Price Trend Analysis (2024+)",
)


# %%
