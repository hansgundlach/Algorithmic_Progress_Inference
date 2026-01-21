"""
Price Bucket Frontier Analysis: Time Trends Within Fixed Price Ranges

This script analyzes time trends for frontier (best ever) models within fixed price buckets.
Key question: Do cheap and expensive models show different rates of algorithmic progress?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy import stats

sns.set_style('whitegrid')


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
    df['Release Date'] = pd.to_datetime(df['Release Date'])

    # Clean score column
    df['Score'] = df[score_col].astype(str).str.replace('%', '').astype(float)

    # Clean price
    df['Price'] = df['Benchmark Cost USD'].astype(str).str.replace('[$,]', '', regex=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Filter
    df_clean = df[['Model', 'Release Date', 'Score', 'Price']].dropna()
    df_clean = df_clean[(df_clean['Price'] > 0) & (df_clean['Score'] > 0)].copy()

    # Add transformations
    df_clean['log_Price'] = np.log10(df_clean['Price'])
    df_clean['Score_logit'] = logit(df_clean['Score'] / 100)

    # Add time variables
    min_date_ordinal = df_clean['Release Date'].min().toordinal()
    df_clean['Date_Ordinal'] = df_clean['Release Date'].map(datetime.toordinal)
    df_clean['Years_Since_Start'] = (df_clean['Date_Ordinal'] - min_date_ordinal) / 365.25

    df_clean['Benchmark'] = benchmark_name

    return df_clean


def calculate_performance_frontier(df):
    """
    Calculate performance frontier models (strictly better than ALL previous models).
    Only includes models that beat the best previous performance, regardless of price.
    """
    df_sorted = df.copy().sort_values('Release Date')
    frontier_indices = []

    current_best = -np.inf

    for idx, row in df_sorted.iterrows():
        if row['Score_logit'] > current_best:
            frontier_indices.append(idx)
            current_best = row['Score_logit']

    return df_sorted.loc[frontier_indices].copy()


def run_regression_on_bucket(df, bucket_name, benchmark_name):
    """Run time trend regression on frontier models in a price bucket"""
    if len(df) < 3:
        return None

    # Calculate frontier
    df_frontier = calculate_performance_frontier(df)

    if len(df_frontier) < 2:  # Need at least 2 points for regression
        return None

    X = df_frontier['Years_Since_Start'].values.reshape(-1, 1)
    y = df_frontier['Score_logit'].values

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
    if n > 2:
        adj_r2 = 1 - (1 - r_squared) * (n - 1) / (n - 2)
    else:
        adj_r2 = r_squared

    return {
        'Benchmark': benchmark_name,
        'Price Bucket': bucket_name,
        'N (All)': len(df),
        'N (Frontier)': len(df_frontier),
        'Time Coef (logits/yr)': model.coef_[0],
        'Std Error': se,
        'p-value': p_value,
        '95% CI Lower': ci_lower,
        '95% CI Upper': ci_upper,
        'R²': r_squared,
        'Adj R²': adj_r2,
        'Mean Score (%)': df_frontier['Score'].mean(),
        'Score Range': f"{df_frontier['Score'].min():.1f}-{df_frontier['Score'].max():.1f}%",
        'Price Range': f"${df['Price'].min():.2f}-${df['Price'].max():.2f}"
    }


def main():
    print("="*120)
    print("PRICE BUCKET FRONTIER ANALYSIS: TIME TRENDS WITHIN FIXED PRICE RANGES")
    print("="*120)
    print()

    # Load data
    print("Loading benchmark data...")
    df_gpqa = load_and_prepare_data('data/price_reduction_models.csv', 'epoch_gpqa', 'GPQA-D')
    df_swe = load_and_prepare_data('data/swe_price_reduction_models.csv', 'epoch_swe', 'SWE-Bench')
    df_aime = load_and_prepare_data('data/aime_price_reduction_models.csv', 'oneshot_AIME', 'AIME')

    print(f"GPQA-D: {len(df_gpqa)} models")
    print(f"SWE-Bench: {len(df_swe)} models")
    print(f"AIME: {len(df_aime)} models")
    print()

    # Define price buckets
    buckets = [
        ('$0.00-$0.05', 0.0, 0.05),
        ('$0.05-$0.10', 0.05, 0.10),
        ('$0.10-$0.50', 0.10, 0.50),
        ('$0.50-$1.00', 0.50, 1.00),
        ('$1.00-$2.00', 1.00, 2.00),
        ('$2.00-$5.00', 2.00, 5.00),
        ('$5.00-$10.00', 5.00, 10.00),
        ('$10.00-$20.00', 10.00, 20.00),
        ('$20.00-$50.00', 20.00, 50.00),
        ('$50.00+', 50.00, float('inf'))
    ]

    results = []

    # Run analysis for each benchmark and bucket
    for df, name in [(df_gpqa, 'GPQA-D'), (df_swe, 'SWE-Bench'), (df_aime, 'AIME')]:
        for bucket_name, lower, upper in buckets:
            df_bucket = df[(df['Price'] >= lower) & (df['Price'] < upper)].copy()
            result = run_regression_on_bucket(df_bucket, bucket_name, name)
            if result is not None:
                results.append(result)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("No results generated. Check data and bucket definitions.")
        return

    # Print summary table
    print("\n" + "="*120)
    print("FRONTIER TIME TRENDS WITHIN FIXED PRICE BUCKETS")
    print("Model: logit(score) ~ years_since_start (for frontier models only)")
    print("="*120 + "\n")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 220)
    pd.set_option('display.precision', 4)

    # Sort for better readability
    results_df_sorted = results_df.sort_values(['Benchmark', 'Price Bucket'])
    print(results_df_sorted.to_string(index=False))

    # Save results
    results_df.to_csv('results/price_bucket_frontier_regressions.csv', index=False)
    print("\n\nSaved to: results/price_bucket_frontier_regressions.csv")

    # Create visualization
    print("\nCreating visualizations...")

    # Plot 1: Time coefficient by price bucket
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    benchmarks = ['GPQA-D', 'SWE-Bench', 'AIME']
    colors = ['red', 'blue', 'green']

    for idx, (benchmark, color) in enumerate(zip(benchmarks, colors)):
        ax = axes[idx]

        # Get data for this benchmark
        bench_data = results_df[results_df['Benchmark'] == benchmark].copy()

        if len(bench_data) == 0:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{benchmark}\nFrontier Progress by Price Bucket',
                        fontsize=12, fontweight='bold')
            continue

        # Create x-axis positions
        x_pos = np.arange(len(bench_data))

        # Plot bars
        bars = ax.bar(x_pos, bench_data['Time Coef (logits/yr)'],
                     color=color, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add error bars (95% CI)
        errors_lower = bench_data['Time Coef (logits/yr)'] - bench_data['95% CI Lower']
        errors_upper = bench_data['95% CI Upper'] - bench_data['Time Coef (logits/yr)']
        ax.errorbar(x_pos, bench_data['Time Coef (logits/yr)'],
                   yerr=[errors_lower, errors_upper],
                   fmt='none', ecolor='black', capsize=5, capthick=2, alpha=0.6)

        # Add sample size annotations
        for i, (pos, row) in enumerate(zip(x_pos, bench_data.iterrows())):
            _, row = row
            ax.text(pos, row['Time Coef (logits/yr)'] + 0.1,
                   f"N={row['N (Frontier)']}",
                   ha='center', fontsize=8, fontweight='bold')

        # Add horizontal line at zero
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Format x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bench_data['Price Bucket'], rotation=45, ha='right', fontsize=9)

        ax.set_xlabel('Price Bucket', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frontier Time Coefficient (logits/yr)', fontsize=11, fontweight='bold')
        ax.set_title(f'{benchmark}\nFrontier Progress by Price Bucket',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figures/price_bucket_frontier_trends.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/price_bucket_frontier_trends.png")
    plt.show()

    # Plot 2: Sample sizes by bucket
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (benchmark, color) in enumerate(zip(benchmarks, colors)):
        ax = axes[idx]

        bench_data = results_df[results_df['Benchmark'] == benchmark].copy()

        if len(bench_data) == 0:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{benchmark}\nSample Sizes', fontsize=12, fontweight='bold')
            continue

        x_pos = np.arange(len(bench_data))

        # Plot two bars: All models and Frontier models
        width = 0.35
        ax.bar(x_pos - width/2, bench_data['N (All)'], width,
              label='All models in bucket', color=color, alpha=0.5, edgecolor='black')
        ax.bar(x_pos + width/2, bench_data['N (Frontier)'], width,
              label='Frontier models', color=color, alpha=0.9, edgecolor='black')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(bench_data['Price Bucket'], rotation=45, ha='right', fontsize=9)

        ax.set_xlabel('Price Bucket', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Models', fontsize=11, fontweight='bold')
        ax.set_title(f'{benchmark}\nSample Sizes by Bucket', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figures/price_bucket_sample_sizes.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/price_bucket_sample_sizes.png")
    plt.show()

    print("\nAnalysis complete!")
    print("\nKEY INSIGHTS:")
    print("="*120)
    print("Plot 1 (Frontier Progress): Time coefficient for frontier (best ever) models in each price bucket")
    print("  - Shows whether cheap vs expensive models have different rates of algorithmic progress")
    print("  - High coefficients = rapid frontier advancement within that price range")
    print("  - Low/negative coefficients = stagnant frontier (no improvement over time)")
    print()
    print("Plot 2 (Sample Sizes): Shows how many models and frontier models exist in each bucket")
    print()
    print("INTERPRETATION:")
    print("  - If high-price buckets show faster progress: Breakthroughs require expensive models")
    print("  - If all buckets show similar progress: Algorithmic advances are price-independent")
    print("  - If low-price buckets show faster progress: Innovation happening in cheaper models")
    print("="*120)


if __name__ == '__main__':
    main()
