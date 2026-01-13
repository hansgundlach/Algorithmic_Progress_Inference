"""
Training Compute Analysis for GPQA-D, AIME, and SWE-Bench

This script analyzes how much performance improvement is due to:
1. Increased training compute (scaling)
2. Algorithmic/architectural improvements (time after controlling for compute)

For each benchmark, we run:
- Model 1: logit(score) ~ time (no compute control)
- Model 2: logit(score) ~ time + log10(training_compute) (with compute control)

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

def run_regression_analysis(df, group_name):
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

    return {
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

# Load GPQA Diamond data
gpqa_df = pd.read_csv('data/gpqa_diamond.csv')
gpqa_df = gpqa_df[['Model version', 'mean_score', 'Release date', 'Training compute (FLOP)']].copy()
gpqa_df.columns = ['Model', 'Score', 'Release_Date', 'Training_Compute']

# Clean data
gpqa_df['Release_Date'] = pd.to_datetime(gpqa_df['Release_Date'])
gpqa_df['Training_Compute'] = pd.to_numeric(gpqa_df['Training_Compute'], errors='coerce')

# Filter to valid data
gpqa_clean = gpqa_df.dropna(subset=['Score', 'Release_Date', 'Training_Compute']).copy()
gpqa_clean = gpqa_clean[gpqa_clean['Score'] > 0].copy()
gpqa_clean = gpqa_clean[gpqa_clean['Training_Compute'] > 0].copy()

# Add transformations
gpqa_clean['Score_logit'] = logit(gpqa_clean['Score'])
gpqa_clean['log_Training_Compute'] = np.log10(gpqa_clean['Training_Compute'])

# Add time variables
min_date_ordinal = gpqa_clean['Release_Date'].min().toordinal()
gpqa_clean['Date_Ordinal'] = gpqa_clean['Release_Date'].map(datetime.toordinal)
gpqa_clean['Years_Since_Start'] = (gpqa_clean['Date_Ordinal'] - min_date_ordinal) / 365.25

print(f"\nGPQA-D models with complete data: {len(gpqa_clean)}")
print(f"Date range: {gpqa_clean['Release_Date'].min()} to {gpqa_clean['Release_Date'].max()}")

# Identify frontier models
gpqa_frontier = identify_frontier_models(gpqa_clean)
print(f"GPQA-D Frontier models: {len(gpqa_frontier)}")

# Run regression analysis
results_gpqa_all = run_regression_analysis(gpqa_clean, 'All Models')
results_gpqa_frontier = run_regression_analysis(gpqa_frontier, 'Frontier Models')

# Create summary table
gpqa_summary = pd.DataFrame([
    {k: v for k, v in results_gpqa_all.items() if k not in ['model1', 'model2', 'df']},
    {k: v for k, v in results_gpqa_frontier.items() if k not in ['model1', 'model2', 'df']}
])

print("\nGPQA-D Results:")
print(gpqa_summary.to_string(index=False))
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

# Load merged data with training compute
merged_df = pd.read_csv('data/merged_with_training_compute.csv')

# Select AIME data (using oneshot_AIME for better coverage)
aime_df = merged_df[['Model', 'Release Date', 'oneshot_AIME', 'Training_Compute_FLOP']].copy()
aime_df.columns = ['Model', 'Release_Date', 'Score', 'Training_Compute']

# Clean data
aime_df['Release_Date'] = pd.to_datetime(aime_df['Release_Date'], errors='coerce')
aime_df['Training_Compute'] = pd.to_numeric(aime_df['Training_Compute'], errors='coerce')
# Handle percentage strings
aime_df['Score'] = aime_df['Score'].astype(str).str.replace('%', '').str.strip()
aime_df['Score'] = pd.to_numeric(aime_df['Score'], errors='coerce')

# Filter to valid data - be very explicit
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

    # Add time variables
    min_date_ordinal = aime_clean['Release_Date'].min().toordinal()
    aime_clean['Date_Ordinal'] = aime_clean['Release_Date'].map(datetime.toordinal)
    aime_clean['Years_Since_Start'] = (aime_clean['Date_Ordinal'] - min_date_ordinal) / 365.25

    print(f"Date range: {aime_clean['Release_Date'].min()} to {aime_clean['Release_Date'].max()}")

    # Identify frontier models
    aime_frontier = identify_frontier_models(aime_clean)
    print(f"AIME Frontier models: {len(aime_frontier)}")

    # Run regression analysis
    results_aime_all = run_regression_analysis(aime_clean, 'All Models')
    results_aime_frontier = run_regression_analysis(aime_frontier, 'Frontier Models')

if results_aime_all is not None and results_aime_frontier is not None:
    # Create summary table
    aime_summary = pd.DataFrame([
        {k: v for k, v in results_aime_all.items() if k not in ['model1', 'model2', 'df']},
        {k: v for k, v in results_aime_frontier.items() if k not in ['model1', 'model2', 'df']}
    ])

    print("\nAIME Results:")
    print(aime_summary.to_string(index=False))
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

# Select SWE-Bench data
swe_df = merged_df[['Model', 'Release Date', 'epoch_swe', 'Training_Compute_FLOP']].copy()
swe_df.columns = ['Model', 'Release_Date', 'Score', 'Training_Compute']

# Clean data
swe_df['Release_Date'] = pd.to_datetime(swe_df['Release_Date'], errors='coerce')
swe_df['Training_Compute'] = pd.to_numeric(swe_df['Training_Compute'], errors='coerce')
# Handle percentage strings
swe_df['Score'] = swe_df['Score'].astype(str).str.replace('%', '').str.strip()
swe_df['Score'] = pd.to_numeric(swe_df['Score'], errors='coerce')

# Filter to valid data - be very explicit
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

    # Add time variables
    min_date_ordinal = swe_clean['Release_Date'].min().toordinal()
    swe_clean['Date_Ordinal'] = swe_clean['Release_Date'].map(datetime.toordinal)
    swe_clean['Years_Since_Start'] = (swe_clean['Date_Ordinal'] - min_date_ordinal) / 365.25

    print(f"Date range: {swe_clean['Release_Date'].min()} to {swe_clean['Release_Date'].max()}")

    # Identify frontier models
    swe_frontier = identify_frontier_models(swe_clean)
    print(f"SWE-Bench Frontier models: {len(swe_frontier)}")

    # Run regression analysis
    results_swe_all = run_regression_analysis(swe_clean, 'All Models')
    results_swe_frontier = run_regression_analysis(swe_frontier, 'Frontier Models')

if results_swe_all is not None and results_swe_frontier is not None:
    # Create summary table
    swe_summary = pd.DataFrame([
        {k: v for k, v in results_swe_all.items() if k not in ['model1', 'model2', 'df']},
        {k: v for k, v in results_swe_frontier.items() if k not in ['model1', 'model2', 'df']}
    ])

    print("\nSWE-Bench Results:")
    print(swe_summary.to_string(index=False))
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
    (results_aime_all, 'AIME'),
    (results_aime_frontier, 'AIME'),
    (results_swe_all, 'SWE-Bench'),
    (results_swe_frontier, 'SWE-Bench')
]:
    if result_dict is not None:
        result_copy = {k: v for k, v in result_dict.items() if k not in ['model1', 'model2', 'df']}
        result_copy['Benchmark'] = benchmark
        all_results.append(result_copy)

comparison_df = pd.DataFrame(all_results)

# Reorder columns
column_order = ['Benchmark', 'Group', 'N', 'Time Coef (no control)', 'p-value', 'R² (no control)',
                'Time Coef (w/ compute)', 'p-value (time)', 'Compute Coef', 'p-value (compute)',
                'R² (w/ compute)', 'Coef Δ', 'Coef Δ (%)']
comparison_df = comparison_df[column_order]

print("\nCross-Benchmark Comparison:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.precision', 4)
print(comparison_df.to_string(index=False))

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
