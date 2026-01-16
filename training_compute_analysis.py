"""
Training Compute Analysis for GPQA-D, AIME, and SWE-Bench

This script analyzes how much performance improvement is due to:
1. Increased training compute (scaling)
2. Algorithmic/architectural improvements (time after controlling for compute)
3. Price efficiency (benchmark cost after controlling for compute)

For each benchmark, we run:
- Model 1: logit(score) ~ time (no controls)
- Model 2: logit(score) ~ time + log10(training_compute) (with compute control)
- Model 3: logit(score) ~ time + log10(training_compute) + log10(price) (with both controls)

Groups analyzed:
- All models: Every model in the dataset
- Frontier models: Models that achieved better performance than any previous model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy import stats
import os

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

sns.set_style('whitegrid')

print("="*80)
print("TRAINING COMPUTE ANALYSIS")
print("="*80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def logit(p):
    """Convert probability to logit scale"""
    p_clipped = np.clip(p, 0.001, 0.999)
    return np.log(p_clipped / (1 - p_clipped))

def inverse_logit(logit_val):
    """Convert logit back to probability"""
    return 1 / (1 + np.exp(-logit_val))

def identify_frontier_models(df):
    """Identify frontier models (better than all previous)"""
    df_sorted = df.sort_values('Release_Date').copy()
    frontier_indices = []
    max_score_so_far = -np.inf

    for idx, row in df_sorted.iterrows():
        if row['Score'] > max_score_so_far:
            frontier_indices.append(idx)
            max_score_so_far = row['Score']

    return df.loc[frontier_indices].copy()

def identify_pareto_models(df):
    """Identify Pareto frontier models (better OR cheaper than all previous)

    A model is on the Pareto frontier if no other model dominates it.
    Model A dominates Model B if A has both:
    - Better or equal performance (score >= B's score)
    - Lower or equal price (price <= B's price)
    with at least one strict inequality.
    """
    # Need both score and price
    df_valid = df[pd.notna(df['Benchmark_Price']) & (df['Benchmark_Price'] > 0)].copy()

    if len(df_valid) == 0:
        return pd.DataFrame()

    pareto_indices = []

    for idx, row in df_valid.iterrows():
        # Check if this model is dominated by any other model
        is_dominated = False
        for other_idx, other_row in df_valid.iterrows():
            if other_idx == idx:
                continue

            # Check if other model dominates this one
            better_or_equal_score = other_row['Score'] >= row['Score']
            cheaper_or_equal_price = other_row['Benchmark_Price'] <= row['Benchmark_Price']
            at_least_one_strict = (other_row['Score'] > row['Score']) or (other_row['Benchmark_Price'] < row['Benchmark_Price'])

            if better_or_equal_score and cheaper_or_equal_price and at_least_one_strict:
                is_dominated = True
                break

        if not is_dominated:
            pareto_indices.append(idx)

    return df.loc[pareto_indices].copy()

def run_regression_analysis(df, group_name, benchmark_name=''):
    """Run regression analysis for a group"""

    # Prepare data
    X_time = df['Years_Since_Start'].values.reshape(-1, 1)
    X_time_compute = df[['Years_Since_Start', 'log_Training_Compute']].values
    y_logit = df['Score_logit'].values
    n = len(y_logit)

    # Check for minimum sample size
    if n < 3:
        print(f"  WARNING: {group_name} has only {n} models - need at least 3 for regression")
        return None

    # Model 1: Without compute control
    model1 = LinearRegression().fit(X_time, y_logit)
    r2_1 = model1.score(X_time, y_logit)

    # Model 1 statistics
    residuals1 = y_logit - model1.predict(X_time)
    mse1 = np.sum(residuals1**2) / (n - 2)
    X_centered = X_time - np.mean(X_time)
    se1 = np.sqrt(mse1 / np.sum(X_centered**2))
    t_stat1 = model1.coef_[0] / se1
    p_value1 = 2 * (1 - stats.t.cdf(np.abs(t_stat1), n - 2))

    # Model 2: With compute control
    model2 = LinearRegression().fit(X_time_compute, y_logit)
    r2_2 = model2.score(X_time_compute, y_logit)

    # Model 2 statistics
    residuals2 = y_logit - model2.predict(X_time_compute)
    mse2 = np.sum(residuals2**2) / (n - 3)
    XtX_inv = np.linalg.inv(X_time_compute.T @ X_time_compute)
    se_coefs2 = np.sqrt(np.diag(XtX_inv) * mse2)
    t_stat2_time = model2.coef_[0] / se_coefs2[0]
    t_stat2_compute = model2.coef_[1] / se_coefs2[1]
    p_value2_time = 2 * (1 - stats.t.cdf(np.abs(t_stat2_time), n - 3))
    p_value2_compute = 2 * (1 - stats.t.cdf(np.abs(t_stat2_compute), n - 3))

    # Calculate coefficient change
    coef_change = model2.coef_[0] - model1.coef_[0]
    pct_change = (coef_change / model1.coef_[0]) * 100

    result = {
        'Group': group_name,
        'N': n,
        'Time Coef (no control)': model1.coef_[0],
        'p-value': p_value1,
        'R² (no control)': r2_1,
        'Time Coef (w/ compute)': model2.coef_[0],
        'p-value (time)': p_value2_time,
        'Compute Coef': model2.coef_[1],
        'p-value (compute)': p_value2_compute,
        'R² (w/ compute)': r2_2,
        'Coef Δ': coef_change,
        'Coef Δ (%)': pct_change,
        'model1': model1,
        'model2': model2,
        'df': df
    }

    # Model 3: With both compute and price control (if price data available)
    price_available = df['log_Price'].notna().sum()
    if price_available >= n and n >= 4:  # Need all models to have price data
        X_time_compute_price = df[['Years_Since_Start', 'log_Training_Compute', 'log_Price']].values

        try:
            model3 = LinearRegression().fit(X_time_compute_price, y_logit)
            r2_3 = model3.score(X_time_compute_price, y_logit)

            # Model 3 statistics
            residuals3 = y_logit - model3.predict(X_time_compute_price)
            mse3 = np.sum(residuals3**2) / (n - 4)
            XtX_inv3 = np.linalg.inv(X_time_compute_price.T @ X_time_compute_price)
            se_coefs3 = np.sqrt(np.diag(XtX_inv3) * mse3)
            t_stat3_time = model3.coef_[0] / se_coefs3[0]
            t_stat3_compute = model3.coef_[1] / se_coefs3[1]
            t_stat3_price = model3.coef_[2] / se_coefs3[2]
            p_value3_time = 2 * (1 - stats.t.cdf(np.abs(t_stat3_time), n - 4))
            p_value3_compute = 2 * (1 - stats.t.cdf(np.abs(t_stat3_compute), n - 4))
            p_value3_price = 2 * (1 - stats.t.cdf(np.abs(t_stat3_price), n - 4))

            coef_change3 = model3.coef_[0] - model1.coef_[0]
            pct_change3 = (coef_change3 / model1.coef_[0]) * 100

            result['Time Coef (w/ compute+price)'] = model3.coef_[0]
            result['p-value (time) 3'] = p_value3_time
            result['Compute Coef 3'] = model3.coef_[1]
            result['p-value (compute) 3'] = p_value3_compute
            result['Price Coef'] = model3.coef_[2]
            result['p-value (price)'] = p_value3_price
            result['R² (w/ compute+price)'] = r2_3
            result['Coef Δ 3'] = coef_change3
            result['Coef Δ (%) 3'] = pct_change3
            result['model3'] = model3
        except Exception as e:
            print(f"  WARNING: Model 3 failed for {group_name}: {e}")

    return result

def create_binscatter_plot(results_all, results_frontier, benchmark_name, filename):
    """Create binscatter-style visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    datasets = [
        (results_all, 'All Models'),
        (results_frontier, 'Frontier Models')
    ]

    for idx, (results, group_name) in enumerate(datasets):
        ax = axes[idx]
        df = results['df']

        # Prepare data
        X_time = df['Years_Since_Start'].values.reshape(-1, 1)
        X_compute = df['log_Training_Compute'].values.reshape(-1, 1)
        y_logit = df['Score_logit'].values

        # Calculate compute-residualized points
        model_compute_only = LinearRegression().fit(X_compute, y_logit)
        compute_effect = model_compute_only.predict(X_compute)
        y_residualized = y_logit - compute_effect + np.mean(y_logit)

        # Plot RAW data (RED) - IN LOGITS
        ax.scatter(df['Release_Date'], y_logit, alpha=0.6, s=80,
                  color='red', edgecolors='black', linewidth=0.5,
                  label='Observed (unadjusted)', zorder=3)

        # Red trend line
        dates_range = pd.date_range(df['Release_Date'].min(), df['Release_Date'].max(), periods=100)
        years_range = (dates_range - df['Release_Date'].min()).days / 365.25
        pred1_logit = results['model1'].predict(years_range.values.reshape(-1, 1))
        ax.plot(dates_range, pred1_logit, '-', color='red', linewidth=2.5,
               label=f'Observed trend ({results["Time Coef (no control)"]:.2f} logits/yr)',
               alpha=0.8, zorder=2)

        # Plot COMPUTE-ADJUSTED data (BLUE) - IN LOGITS
        ax.scatter(df['Release_Date'], y_residualized, alpha=0.6, s=80,
                  color='blue', edgecolors='black', linewidth=0.5,
                  label='Compute-adjusted', zorder=3)

        # Blue trend line
        model_residualized = LinearRegression().fit(X_time, y_residualized)
        pred2_logit = model_residualized.predict(years_range.values.reshape(-1, 1))
        ax.plot(dates_range, pred2_logit, '-', color='blue', linewidth=2.5,
               label=f'Compute-adjusted trend ({results["Time Coef (w/ compute)"]:.2f} logits/yr)',
               alpha=0.8, zorder=2)

        ax.set_xlabel('Release Date', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{benchmark_name} Score (logits)', fontsize=11, fontweight='bold')
        ax.set_title(f'{benchmark_name} {group_name}\n(N={results["N"]})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3, zorder=1)

        # Add stats box
        stats_text = f'Compute effect: {results["Compute Coef"]:.2f} logits\n'
        stats_text += f'Coef Δ: {results["Coef Δ (%)"]:.1f}%'

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved figure: figures/{filename}")

# ============================================================================
# GPQA DIAMOND ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("1. GPQA DIAMOND ANALYSIS")
print("="*80)

# Load GPQA data from price_reduction_models.csv
gpqa_df = pd.read_csv('data/price_reduction_models.csv')
print(f"Loaded {len(gpqa_df)} models from price_reduction_models.csv")

# Extract relevant columns
gpqa_df = gpqa_df[['Model', 'GPQA Diamond (Scientific Reasoning)', 'Release Date',
                    'Training_Compute_FLOP', 'Benchmark Cost USD']].copy()
gpqa_df.columns = ['Model', 'Score', 'Release_Date', 'Training_Compute', 'Benchmark_Price']

# Clean data
gpqa_df['Release_Date'] = pd.to_datetime(gpqa_df['Release_Date'], errors='coerce')
gpqa_df['Training_Compute'] = pd.to_numeric(gpqa_df['Training_Compute'], errors='coerce')
gpqa_df['Benchmark_Price'] = pd.to_numeric(gpqa_df['Benchmark_Price'], errors='coerce')
# Handle percentage strings for score
gpqa_df['Score'] = gpqa_df['Score'].astype(str).str.replace('%', '').str.strip()
gpqa_df['Score'] = pd.to_numeric(gpqa_df['Score'], errors='coerce')

# Filter to valid data
gpqa_clean = gpqa_df.dropna(subset=['Score', 'Release_Date', 'Training_Compute']).copy()
gpqa_clean = gpqa_clean[gpqa_clean['Score'] > 0].copy()
gpqa_clean = gpqa_clean[gpqa_clean['Training_Compute'] > 0].copy()

# Convert scores to proportion if they're percentages (>1)
if gpqa_clean['Score'].max() > 1:
    gpqa_clean['Score'] = gpqa_clean['Score'] / 100.0

# Add transformations
gpqa_clean['Score_logit'] = logit(gpqa_clean['Score'])
gpqa_clean['log_Training_Compute'] = np.log10(gpqa_clean['Training_Compute'])

# Add log price
gpqa_clean['log_Price'] = np.nan
price_valid = (gpqa_clean['Benchmark_Price'] > 0) & pd.notna(gpqa_clean['Benchmark_Price'])
gpqa_clean.loc[price_valid, 'log_Price'] = np.log10(gpqa_clean.loc[price_valid, 'Benchmark_Price'])

# Add time variables
min_date_ordinal = gpqa_clean['Release_Date'].min().toordinal()
gpqa_clean['Date_Ordinal'] = gpqa_clean['Release_Date'].map(datetime.toordinal)
gpqa_clean['Years_Since_Start'] = (gpqa_clean['Date_Ordinal'] - min_date_ordinal) / 365.25

print(f"\nGPQA-D models with complete data: {len(gpqa_clean)}")
print(f"Date range: {gpqa_clean['Release_Date'].min()} to {gpqa_clean['Release_Date'].max()}")
print(f"Models with benchmark price data: {price_valid.sum()}")

# Identify frontier models
gpqa_frontier = identify_frontier_models(gpqa_clean)
print(f"GPQA-D Frontier models: {len(gpqa_frontier)}")

# Identify Pareto frontier models
gpqa_pareto = identify_pareto_models(gpqa_clean)
print(f"GPQA-D Pareto frontier models: {len(gpqa_pareto)}")

# Run regression analysis
results_gpqa_all = run_regression_analysis(gpqa_clean, 'All Models')
results_gpqa_frontier = run_regression_analysis(gpqa_frontier, 'Frontier Models')
results_gpqa_pareto = run_regression_analysis(gpqa_pareto, 'Pareto Frontier') if len(gpqa_pareto) >= 3 else None

# Create summary table
summary_rows = [
    {k: v for k, v in results_gpqa_all.items() if k not in ['model1', 'model2', 'model3', 'df']},
    {k: v for k, v in results_gpqa_frontier.items() if k not in ['model1', 'model2', 'model3', 'df']}
]
if results_gpqa_pareto is not None:
    summary_rows.append({k: v for k, v in results_gpqa_pareto.items() if k not in ['model1', 'model2', 'model3', 'df']})

gpqa_summary = pd.DataFrame(summary_rows)

print("\nGPQA-D Results:")
print(f"{'Group':<17} {'N':>3} | Model 1 (Time) | Model 2 (+ Compute) | Model 3 (+ Price)")
print("-" * 87)
for _, row in gpqa_summary.iterrows():
    m1 = f"{row['Time Coef (no control)']:+.2f} (R²={row['R² (no control)']:.2f})"
    m2 = f"{row['Time Coef (w/ compute)']:+.2f} (R²={row['R² (w/ compute)']:.2f})"
    if 'Time Coef (w/ compute+price)' in row and pd.notna(row['Time Coef (w/ compute+price)']):
        m3 = f"{row['Time Coef (w/ compute+price)']:+.2f} (R²={row['R² (w/ compute+price)']:.2f})"
    else:
        m3 = "N/A"
    print(f"{row['Group']:<17} {int(row['N']):3d} | {m1:14s} | {m2:19s} | {m3:17s}")

# Print detailed Model 3 coefficients
if 'Time Coef (w/ compute+price)' in results_gpqa_all:
    print("\nModel 3 Coefficients (All Models):")
    print(f"  Time: {results_gpqa_all['Time Coef (w/ compute+price)']:+.3f} logits/yr ({results_gpqa_all['Coef Δ (%) 3']:+.1f}% vs baseline)")
    print(f"  Compute: {results_gpqa_all['Compute Coef 3']:+.3f} logits/log10(FLOP)")
    print(f"  Price: {results_gpqa_all['Price Coef']:+.3f} logits/log10(benchmark_cost)")
if 'Time Coef (w/ compute+price)' in results_gpqa_frontier:
    print("\nModel 3 Coefficients (Frontier):")
    print(f"  Time: {results_gpqa_frontier['Time Coef (w/ compute+price)']:+.3f} logits/yr ({results_gpqa_frontier['Coef Δ (%) 3']:+.1f}% vs baseline)")
    print(f"  Compute: {results_gpqa_frontier['Compute Coef 3']:+.3f} logits/log10(FLOP)")
    print(f"  Price: {results_gpqa_frontier['Price Coef']:+.3f} logits/log10(benchmark_cost)")
if results_gpqa_pareto is not None and 'Time Coef (w/ compute+price)' in results_gpqa_pareto:
    print("\nModel 3 Coefficients (Pareto Frontier):")
    print(f"  Time: {results_gpqa_pareto['Time Coef (w/ compute+price)']:+.3f} logits/yr ({results_gpqa_pareto['Coef Δ (%) 3']:+.1f}% vs baseline)")
    print(f"  Compute: {results_gpqa_pareto['Compute Coef 3']:+.3f} logits/log10(FLOP)")
    print(f"  Price: {results_gpqa_pareto['Price Coef']:+.3f} logits/log10(benchmark_cost)")

gpqa_summary.to_csv('results/gpqa_training_compute_regression.csv', index=False)

# Create visualization
create_binscatter_plot(results_gpqa_all, results_gpqa_frontier, 'GPQA-D',
                       'gpqa_binscatter_compute_logits.png')

# ============================================================================
# AIME ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("2. AIME ANALYSIS")
print("="*80)

# Load AIME data from aime_price_reduction_models.csv
aime_df = pd.read_csv('data/aime_price_reduction_models.csv')
print(f"Loaded {len(aime_df)} models from aime_price_reduction_models.csv")

# Extract relevant columns
aime_df = aime_df[['Model', 'oneshot_AIME', 'Release Date',
                    'Training_Compute_FLOP', 'Benchmark Cost USD']].copy()
aime_df.columns = ['Model', 'Score', 'Release_Date', 'Training_Compute', 'Benchmark_Price']

# Clean data
aime_df['Release_Date'] = pd.to_datetime(aime_df['Release_Date'], errors='coerce')
aime_df['Training_Compute'] = pd.to_numeric(aime_df['Training_Compute'], errors='coerce')
aime_df['Benchmark_Price'] = pd.to_numeric(aime_df['Benchmark_Price'], errors='coerce')
# Handle percentage strings
aime_df['Score'] = aime_df['Score'].astype(str).str.replace('%', '').str.strip()
aime_df['Score'] = pd.to_numeric(aime_df['Score'], errors='coerce')

# Filter to valid data
aime_clean = aime_df.copy()
aime_clean = aime_clean[pd.notna(aime_clean['Score'])].copy()
aime_clean = aime_clean[pd.notna(aime_clean['Release_Date'])].copy()
aime_clean = aime_clean[pd.notna(aime_clean['Training_Compute'])].copy()
aime_clean = aime_clean[aime_clean['Score'] > 0].copy()
aime_clean = aime_clean[aime_clean['Score'] < 100].copy()  # Percentage values
aime_clean = aime_clean[aime_clean['Training_Compute'] > 0].copy()

# Convert scores from percentage to proportion
aime_clean['Score'] = aime_clean['Score'] / 100.0

print(f"\nAIME models with complete data: {len(aime_clean)}")

if len(aime_clean) == 0:
    print("WARNING: No AIME models with complete data (score, date, and training compute)")
    print("Skipping AIME analysis")
    results_aime_all = None
    results_aime_frontier = None
else:
    # Add transformations
    aime_clean['Score_logit'] = logit(aime_clean['Score'])
    aime_clean['log_Training_Compute'] = np.log10(aime_clean['Training_Compute'])

    # Add log price
    aime_clean['log_Price'] = np.nan
    price_valid = (aime_clean['Benchmark_Price'] > 0) & pd.notna(aime_clean['Benchmark_Price'])
    aime_clean.loc[price_valid, 'log_Price'] = np.log10(aime_clean.loc[price_valid, 'Benchmark_Price'])

    # Add time variables
    min_date_ordinal = aime_clean['Release_Date'].min().toordinal()
    aime_clean['Date_Ordinal'] = aime_clean['Release_Date'].map(datetime.toordinal)
    aime_clean['Years_Since_Start'] = (aime_clean['Date_Ordinal'] - min_date_ordinal) / 365.25

    print(f"Date range: {aime_clean['Release_Date'].min()} to {aime_clean['Release_Date'].max()}")
    print(f"Models with benchmark price data: {price_valid.sum()}")

    # Identify frontier models
    aime_frontier = identify_frontier_models(aime_clean)
    print(f"AIME Frontier models: {len(aime_frontier)}")

    # Identify Pareto frontier models
    aime_pareto = identify_pareto_models(aime_clean)
    print(f"AIME Pareto frontier models: {len(aime_pareto)}")

    # Run regression analysis
    results_aime_all = run_regression_analysis(aime_clean, 'All Models')
    results_aime_frontier = run_regression_analysis(aime_frontier, 'Frontier Models')
    results_aime_pareto = run_regression_analysis(aime_pareto, 'Pareto Frontier') if len(aime_pareto) >= 3 else None

if results_aime_all is not None and results_aime_frontier is not None:
    # Create summary table
    summary_rows = [
        {k: v for k, v in results_aime_all.items() if k not in ['model1', 'model2', 'model3', 'df']},
        {k: v for k, v in results_aime_frontier.items() if k not in ['model1', 'model2', 'model3', 'df']}
    ]
    if results_aime_pareto is not None:
        summary_rows.append({k: v for k, v in results_aime_pareto.items() if k not in ['model1', 'model2', 'model3', 'df']})

    aime_summary = pd.DataFrame(summary_rows)

    print("\nAIME Results:")
    print(f"{'Group':<17} {'N':>3} | Model 1 (Time) | Model 2 (+ Compute) | Model 3 (+ Price)")
    print("-" * 87)
    for _, row in aime_summary.iterrows():
        m1 = f"{row['Time Coef (no control)']:+.2f} (R²={row['R² (no control)']:.2f})"
        m2 = f"{row['Time Coef (w/ compute)']:+.2f} (R²={row['R² (w/ compute)']:.2f})"
        if 'Time Coef (w/ compute+price)' in row and pd.notna(row['Time Coef (w/ compute+price)']):
            m3 = f"{row['Time Coef (w/ compute+price)']:+.2f} (R²={row['R² (w/ compute+price)']:.2f})"
        else:
            m3 = "N/A"
        print(f"{row['Group']:<17} {int(row['N']):3d} | {m1:14s} | {m2:19s} | {m3:17s}")

    # Print detailed Model 3 coefficients
    if 'Time Coef (w/ compute+price)' in results_aime_all:
        print("\nModel 3 Coefficients (All Models):")
        print(f"  Time: {results_aime_all['Time Coef (w/ compute+price)']:+.3f} logits/yr ({results_aime_all['Coef Δ (%) 3']:+.1f}% vs baseline)")
        print(f"  Compute: {results_aime_all['Compute Coef 3']:+.3f} logits/log10(FLOP)")
        print(f"  Price: {results_aime_all['Price Coef']:+.3f} logits/log10(benchmark_cost)")
    if 'Time Coef (w/ compute+price)' in results_aime_frontier:
        print("\nModel 3 Coefficients (Frontier):")
        print(f"  Time: {results_aime_frontier['Time Coef (w/ compute+price)']:+.3f} logits/yr ({results_aime_frontier['Coef Δ (%) 3']:+.1f}% vs baseline)")
        print(f"  Compute: {results_aime_frontier['Compute Coef 3']:+.3f} logits/log10(FLOP)")
        print(f"  Price: {results_aime_frontier['Price Coef']:+.3f} logits/log10(benchmark_cost)")
    if results_aime_pareto is not None and 'Time Coef (w/ compute+price)' in results_aime_pareto:
        print("\nModel 3 Coefficients (Pareto Frontier):")
        print(f"  Time: {results_aime_pareto['Time Coef (w/ compute+price)']:+.3f} logits/yr ({results_aime_pareto['Coef Δ (%) 3']:+.1f}% vs baseline)")
        print(f"  Compute: {results_aime_pareto['Compute Coef 3']:+.3f} logits/log10(FLOP)")
        print(f"  Price: {results_aime_pareto['Price Coef']:+.3f} logits/log10(benchmark_cost)")

    aime_summary.to_csv('results/aime_training_compute_regression.csv', index=False)

    # Create visualization
    create_binscatter_plot(results_aime_all, results_aime_frontier, 'AIME',
                           'aime_binscatter_compute_logits.png')

# ============================================================================
# SWE-BENCH ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("3. SWE-BENCH ANALYSIS")
print("="*80)

# Load SWE data from swe_price_reduction_models.csv
swe_df = pd.read_csv('data/swe_price_reduction_models.csv')
print(f"Loaded {len(swe_df)} models from swe_price_reduction_models.csv")

# Extract relevant columns
swe_df = swe_df[['Model', 'epoch_swe', 'Release Date',
                  'Training_Compute_FLOP', 'Benchmark Cost USD']].copy()
swe_df.columns = ['Model', 'Score', 'Release_Date', 'Training_Compute', 'Benchmark_Price']

# Clean data
swe_df['Release_Date'] = pd.to_datetime(swe_df['Release_Date'], errors='coerce')
swe_df['Training_Compute'] = pd.to_numeric(swe_df['Training_Compute'], errors='coerce')
swe_df['Benchmark_Price'] = pd.to_numeric(swe_df['Benchmark_Price'], errors='coerce')
# Handle percentage strings
swe_df['Score'] = swe_df['Score'].astype(str).str.replace('%', '').str.strip()
swe_df['Score'] = pd.to_numeric(swe_df['Score'], errors='coerce')

# Filter to valid data
swe_clean = swe_df.copy()
swe_clean = swe_clean[pd.notna(swe_clean['Score'])].copy()
swe_clean = swe_clean[pd.notna(swe_clean['Release_Date'])].copy()
swe_clean = swe_clean[pd.notna(swe_clean['Training_Compute'])].copy()
swe_clean = swe_clean[swe_clean['Score'] > 0].copy()
swe_clean = swe_clean[swe_clean['Score'] < 100].copy()  # Percentage values
swe_clean = swe_clean[swe_clean['Training_Compute'] > 0].copy()

# Convert scores from percentage to proportion
swe_clean['Score'] = swe_clean['Score'] / 100.0

print(f"\nSWE-Bench models with complete data: {len(swe_clean)}")

if len(swe_clean) == 0:
    print("WARNING: No SWE-Bench models with complete data (score, date, and training compute)")
    print("Skipping SWE-Bench analysis")
    results_swe_all = None
    results_swe_frontier = None
else:
    # Add transformations
    swe_clean['Score_logit'] = logit(swe_clean['Score'])
    swe_clean['log_Training_Compute'] = np.log10(swe_clean['Training_Compute'])

    # Add log price
    swe_clean['log_Price'] = np.nan
    price_valid = (swe_clean['Benchmark_Price'] > 0) & pd.notna(swe_clean['Benchmark_Price'])
    swe_clean.loc[price_valid, 'log_Price'] = np.log10(swe_clean.loc[price_valid, 'Benchmark_Price'])

    # Add time variables
    min_date_ordinal = swe_clean['Release_Date'].min().toordinal()
    swe_clean['Date_Ordinal'] = swe_clean['Release_Date'].map(datetime.toordinal)
    swe_clean['Years_Since_Start'] = (swe_clean['Date_Ordinal'] - min_date_ordinal) / 365.25

    print(f"Date range: {swe_clean['Release_Date'].min()} to {swe_clean['Release_Date'].max()}")
    print(f"Models with benchmark price data: {price_valid.sum()}")

    # Identify frontier models
    swe_frontier = identify_frontier_models(swe_clean)
    print(f"SWE-Bench Frontier models: {len(swe_frontier)}")

    # Identify Pareto frontier models
    swe_pareto = identify_pareto_models(swe_clean)
    print(f"SWE-Bench Pareto frontier models: {len(swe_pareto)}")

    # Run regression analysis
    results_swe_all = run_regression_analysis(swe_clean, 'All Models')
    results_swe_frontier = run_regression_analysis(swe_frontier, 'Frontier Models')
    results_swe_pareto = run_regression_analysis(swe_pareto, 'Pareto Frontier') if len(swe_pareto) >= 3 else None

if results_swe_all is not None and results_swe_frontier is not None:
    # Create summary table
    summary_rows = [
        {k: v for k, v in results_swe_all.items() if k not in ['model1', 'model2', 'model3', 'df']},
        {k: v for k, v in results_swe_frontier.items() if k not in ['model1', 'model2', 'model3', 'df']}
    ]
    if results_swe_pareto is not None:
        summary_rows.append({k: v for k, v in results_swe_pareto.items() if k not in ['model1', 'model2', 'model3', 'df']})

    swe_summary = pd.DataFrame(summary_rows)

    print("\nSWE-Bench Results:")
    print(f"{'Group':<17} {'N':>3} | Model 1 (Time) | Model 2 (+ Compute) | Model 3 (+ Price)")
    print("-" * 87)
    for _, row in swe_summary.iterrows():
        m1 = f"{row['Time Coef (no control)']:+.2f} (R²={row['R² (no control)']:.2f})"
        m2 = f"{row['Time Coef (w/ compute)']:+.2f} (R²={row['R² (w/ compute)']:.2f})"
        if 'Time Coef (w/ compute+price)' in row and pd.notna(row['Time Coef (w/ compute+price)']):
            m3 = f"{row['Time Coef (w/ compute+price)']:+.2f} (R²={row['R² (w/ compute+price)']:.2f})"
        else:
            m3 = "N/A"
        print(f"{row['Group']:<17} {int(row['N']):3d} | {m1:14s} | {m2:19s} | {m3:17s}")

    # Print detailed Model 3 coefficients
    if 'Time Coef (w/ compute+price)' in results_swe_all:
        print("\nModel 3 Coefficients (All Models):")
        print(f"  Time: {results_swe_all['Time Coef (w/ compute+price)']:+.3f} logits/yr ({results_swe_all['Coef Δ (%) 3']:+.1f}% vs baseline)")
        print(f"  Compute: {results_swe_all['Compute Coef 3']:+.3f} logits/log10(FLOP)")
        print(f"  Price: {results_swe_all['Price Coef']:+.3f} logits/log10(benchmark_cost)")
    if 'Time Coef (w/ compute+price)' in results_swe_frontier:
        print("\nModel 3 Coefficients (Frontier):")
        print(f"  Time: {results_swe_frontier['Time Coef (w/ compute+price)']:+.3f} logits/yr ({results_swe_frontier['Coef Δ (%) 3']:+.1f}% vs baseline)")
        print(f"  Compute: {results_swe_frontier['Compute Coef 3']:+.3f} logits/log10(FLOP)")
        print(f"  Price: {results_swe_frontier['Price Coef']:+.3f} logits/log10(benchmark_cost)")
    if results_swe_pareto is not None and 'Time Coef (w/ compute+price)' in results_swe_pareto:
        print("\nModel 3 Coefficients (Pareto Frontier):")
        print(f"  Time: {results_swe_pareto['Time Coef (w/ compute+price)']:+.3f} logits/yr ({results_swe_pareto['Coef Δ (%) 3']:+.1f}% vs baseline)")
        print(f"  Compute: {results_swe_pareto['Compute Coef 3']:+.3f} logits/log10(FLOP)")
        print(f"  Price: {results_swe_pareto['Price Coef']:+.3f} logits/log10(benchmark_cost)")

    swe_summary.to_csv('results/swe_training_compute_regression.csv', index=False)

    # Create visualization
    create_binscatter_plot(results_swe_all, results_swe_frontier, 'SWE-Bench',
                           'swe_binscatter_compute_logits.png')

# ============================================================================
# CROSS-BENCHMARK COMPARISON
# ============================================================================

print("\n" + "="*80)
print("4. CROSS-BENCHMARK COMPARISON")
print("="*80)

# Combine all results
all_results = []
for result_dict, benchmark in [
    (results_gpqa_all, 'GPQA-D'),
    (results_gpqa_frontier, 'GPQA-D'),
    (results_gpqa_pareto, 'GPQA-D'),
    (results_aime_all, 'AIME'),
    (results_aime_frontier, 'AIME'),
    (results_aime_pareto, 'AIME'),
    (results_swe_all, 'SWE-Bench'),
    (results_swe_frontier, 'SWE-Bench'),
    (results_swe_pareto, 'SWE-Bench')
]:
    if result_dict is not None:
        result_copy = {k: v for k, v in result_dict.items() if k not in ['model1', 'model2', 'model3', 'df']}
        result_copy['Benchmark'] = benchmark
        all_results.append(result_copy)

comparison_df = pd.DataFrame(all_results)

print("\nCross-Benchmark Summary (Time Coefficients):")
print(f"{'Benchmark':<12} {'Group':<17} {'N':>3} | Model 1 | Model 2 | Model 3")
print("-" * 69)
for _, row in comparison_df.iterrows():
    bench = row['Benchmark']
    group = row['Group']
    n = int(row['N'])
    m1 = f"{row['Time Coef (no control)']:+.2f}"
    m2 = f"{row['Time Coef (w/ compute)']:+.2f}"
    if 'Time Coef (w/ compute+price)' in row and pd.notna(row['Time Coef (w/ compute+price)']):
        m3 = f"{row['Time Coef (w/ compute+price)']:+.2f}"
    else:
        m3 = "N/A   "
    print(f"{bench:<12} {group:<17} {n:3d} | {m1:7s} | {m2:7s} | {m3:6s}")

comparison_df.to_csv('results/training_compute_comparison_all_benchmarks.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nResults saved to:")
print("  - results/gpqa_training_compute_regression.csv")
print("  - results/aime_training_compute_regression.csv")
print("  - results/swe_training_compute_regression.csv")
print("  - results/training_compute_comparison_all_benchmarks.csv")
print("\nFigures saved to:")
print("  - figures/gpqa_binscatter_compute_logits.png")
print("  - figures/aime_binscatter_compute_logits.png")
print("  - figures/swe_binscatter_compute_logits.png")
