import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy import stats

# Define logit transformation functions
def logit(p):
    """Convert probability to logit scale"""
    p_clipped = np.clip(p, 0.001, 0.999)
    return np.log(p_clipped / (1 - p_clipped))

def inverse_logit(logit_val):
    """Convert logit back to probability"""
    return 1 / (1 + np.exp(-logit_val))

# Load the data
df = pd.read_csv('data/price_reduction_models.csv')

# Convert Release Date to datetime
df['Release Date'] = pd.to_datetime(df['Release Date'])

# Clean GPQA-D column (epoch_gpqa) - convert percentage strings to floats
df['GPQA_D'] = df['epoch_gpqa'].astype(str).str.replace('%', '').astype(float)

# Clean Benchmark Cost USD - convert string with $ and commas to float
df['Price'] = df['Benchmark Cost USD'].astype(str).str.replace('[$,]', '', regex=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Filter out rows with missing data
df_clean = df[['Model', 'Release Date', 'GPQA_D', 'Price']].dropna()
df_clean = df_clean[df_clean['Price'] > 0]

print(f"Total models with complete data: {len(df_clean)}")
print(f"Date range: {df_clean['Release Date'].min()} to {df_clean['Release Date'].max()}")
print(f"GPQA-D range: {df_clean['GPQA_D'].min():.1f}% to {df_clean['GPQA_D'].max():.1f}%")
print(f"Price range: ${df_clean['Price'].min():.2f} to ${df_clean['Price'].max():.2f}")

# Add logit column (convert percentage to decimal first)
df_clean['GPQA_D_logit'] = logit(df_clean['GPQA_D'] / 100)

# Add log price (for regression)
df_clean['log_Price'] = np.log10(df_clean['Price'])

# Add ordinal date for regression (normalize to years for interpretability)
min_date_ordinal = df_clean['Release Date'].min().toordinal()
df_clean['Date_Ordinal'] = df_clean['Release Date'].map(datetime.toordinal)
df_clean['Years_Since_Start'] = (df_clean['Date_Ordinal'] - min_date_ordinal) / 365.25

print("\nTransformations complete:")
print(f"Logit(GPQA-D) range: {df_clean['GPQA_D_logit'].min():.2f} to {df_clean['GPQA_D_logit'].max():.2f}")
print(f"Log10(Price) range: {df_clean['log_Price'].min():.2f} to {df_clean['log_Price'].max():.2f}")
print(f"Years since start: 0 to {df_clean['Years_Since_Start'].max():.2f}")


def perform_regression_analysis(df, use_pareto_only=True, min_date=None):
    """
    Perform multiple regression analysis of logit(GPQA-D) vs time and price.
    """

    df_work = df.copy()

    # Apply date filter if specified
    if min_date is not None:
        if isinstance(min_date, str):
            min_date = pd.to_datetime(min_date)
        df_work = df_work[df_work['Release Date'] >= min_date]

    # Sort by date
    df_work = df_work.sort_values('Release Date')

    # Get Pareto frontier if requested
    if use_pareto_only:
        # Identify Pareto frontier models at each point in time
        # A model is on the Pareto frontier if no other model dominates it
        # (i.e., no model with both better/equal performance and lower/equal price)
        pareto_indices = []

        for date in df_work['Release Date'].unique():
            # Get all models available at this date
            available_models = df_work[df_work['Release Date'] <= date].copy()

            # Find Pareto frontier at this date
            available_models = available_models.sort_values(['Price', 'GPQA_D'])
            frontier_indices = []

            for i, row in available_models.iterrows():
                # Check if this model is on the Pareto frontier
                dominated = False
                for j in frontier_indices:
                    frontier_row = available_models.loc[j]
                    # A model is dominated if there exists another model with:
                    # 1. Better or equal GPQA-D score AND
                    # 2. Lower or equal price
                    # AND at least one is strictly better
                    if (
                        frontier_row['GPQA_D'] >= row['GPQA_D']
                        and frontier_row['Price'] <= row['Price']
                        and (
                            frontier_row['GPQA_D'] > row['GPQA_D']
                            or frontier_row['Price'] < row['Price']
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
                            row['GPQA_D'] >= frontier_row['GPQA_D']
                            and row['Price'] <= frontier_row['Price']
                            and (
                                row['GPQA_D'] > frontier_row['GPQA_D']
                                or row['Price'] < frontier_row['Price']
                            )
                        ):
                            new_frontier_indices.append(j)
                    frontier_indices = new_frontier_indices + [i]

            # Add models released exactly on this date that are on the frontier
            current_date_models = df_work[df_work['Release Date'] == date]
            for i, row in current_date_models.iterrows():
                if i in frontier_indices:
                    pareto_indices.append(i)

        # Remove duplicates and use for regression
        pareto_indices = list(set(pareto_indices))
        df_analysis = df_work.loc[pareto_indices].copy()
        analysis_type = "Pareto Frontier"
    else:
        df_analysis = df_work.copy()
        analysis_type = "All Models"

    print(f"\n{'='*80}")
    print(f"Analyzing {len(df_analysis)} models ({analysis_type})")
    print(f"Date range: {df_analysis['Release Date'].min().strftime('%Y-%m-%d')} to {df_analysis['Release Date'].max().strftime('%Y-%m-%d')}")
    print(f"GPQA-D range: {df_analysis['GPQA_D'].min():.1f}% to {df_analysis['GPQA_D'].max():.1f}%")
    print(f"{'='*80}\n")

    # Prepare data for regression
    y = df_analysis['GPQA_D_logit'].values
    X_time = df_analysis[['Years_Since_Start']].values
    X_time_price = df_analysis[['Years_Since_Start', 'log_Price']].values

    # Model 1: logit(GPQA-D) ~ time (no price control)
    model1 = LinearRegression().fit(X_time, y)
    y_pred1 = model1.predict(X_time)

    # Model 2: logit(GPQA-D) ~ time + log(price) (controlling for price)
    model2 = LinearRegression().fit(X_time_price, y)
    y_pred2 = model2.predict(X_time_price)

    # Calculate R-squared
    r_squared1 = model1.score(X_time, y)
    r_squared2 = model2.score(X_time_price, y)

    # Calculate statistics for Model 1
    n1 = len(y)
    residuals1 = y - y_pred1
    mse1 = np.sum(residuals1**2) / (n1 - 2)  # n - p - 1, where p=1

    # Standard error for time coefficient (Model 1)
    X_time_centered = X_time - np.mean(X_time)
    se_time1 = np.sqrt(mse1 / np.sum(X_time_centered**2))

    # t-statistic and p-value
    t_stat1 = model1.coef_[0] / se_time1
    p_value1 = 2 * (1 - stats.t.cdf(np.abs(t_stat1), n1 - 2))

    # 95% confidence interval
    t_crit1 = stats.t.ppf(0.975, n1 - 2)
    ci_lower1 = model1.coef_[0] - t_crit1 * se_time1
    ci_upper1 = model1.coef_[0] + t_crit1 * se_time1

    # Calculate statistics for Model 2
    n2 = len(y)
    residuals2 = y - y_pred2
    mse2 = np.sum(residuals2**2) / (n2 - 3)  # n - p - 1, where p=2

    # Standard errors for coefficients (Model 2)
    # Using the formula: SE = sqrt(MSE * (X'X)^-1)
    XtX_inv = np.linalg.inv(X_time_price.T @ X_time_price)
    se_coefs2 = np.sqrt(np.diag(XtX_inv) * mse2)

    se_time2 = se_coefs2[0]
    se_price2 = se_coefs2[1]

    # t-statistics and p-values
    t_stat_time2 = model2.coef_[0] / se_time2
    t_stat_price2 = model2.coef_[1] / se_price2

    p_value_time2 = 2 * (1 - stats.t.cdf(np.abs(t_stat_time2), n2 - 3))
    p_value_price2 = 2 * (1 - stats.t.cdf(np.abs(t_stat_price2), n2 - 3))

    # 95% confidence intervals
    t_crit2 = stats.t.ppf(0.975, n2 - 3)
    ci_lower_time2 = model2.coef_[0] - t_crit2 * se_time2
    ci_upper_time2 = model2.coef_[0] + t_crit2 * se_time2

    # Calculate approximate percentage point improvements
    mean_gpqa = df_analysis['GPQA_D'].mean()
    mean_logit_val = logit(mean_gpqa / 100)

    # Model 1: annual improvement
    future_logit_1 = mean_logit_val + model1.coef_[0]
    future_prob_1 = inverse_logit(future_logit_1) * 100
    annual_pct_1 = future_prob_1 - mean_gpqa

    # Model 2: annual improvement
    future_logit_2 = mean_logit_val + model2.coef_[0]
    future_prob_2 = inverse_logit(future_logit_2) * 100
    annual_pct_2 = future_prob_2 - mean_gpqa

    # Create summary results
    results = []

    # Model 1 results
    results.append({
        'Model': 'Without Price Control',
        'Specification': 'logit(GPQA-D) ~ time',
        'Time_Coefficient': model1.coef_[0],
        'Time_Std_Err': se_time1,
        'Time_p_value': p_value1,
        'Time_CI_Lower': ci_lower1,
        'Time_CI_Upper': ci_upper1,
        'Annual_Improvement_Logits': model1.coef_[0],
        'Annual_Improvement_PctPts': annual_pct_1,
        'Price_Coefficient': np.nan,
        'Price_Std_Err': np.nan,
        'Price_p_value': np.nan,
        'R_Squared': r_squared1,
        'Adj_R_Squared': 1 - (1 - r_squared1) * (n1 - 1) / (n1 - 2),
        'N_Models': len(df_analysis),
        'Mean_GPQA_D': mean_gpqa
    })

    # Model 2 results
    results.append({
        'Model': 'With Price Control',
        'Specification': 'logit(GPQA-D) ~ time + log(price)',
        'Time_Coefficient': model2.coef_[0],
        'Time_Std_Err': se_time2,
        'Time_p_value': p_value_time2,
        'Time_CI_Lower': ci_lower_time2,
        'Time_CI_Upper': ci_upper_time2,
        'Annual_Improvement_Logits': model2.coef_[0],
        'Annual_Improvement_PctPts': annual_pct_2,
        'Price_Coefficient': model2.coef_[1],
        'Price_Std_Err': se_price2,
        'Price_p_value': p_value_price2,
        'R_Squared': r_squared2,
        'Adj_R_Squared': 1 - (1 - r_squared2) * (n2 - 1) / (n2 - 3),
        'N_Models': len(df_analysis),
        'Mean_GPQA_D': mean_gpqa
    })

    results_df = pd.DataFrame(results)

    return {
        'results_df': results_df,
        'model1': model1,
        'model2': model2,
        'data': df_analysis,
        'analysis_type': analysis_type
    }


# Run analysis on Pareto frontier
print("\n" + "="*100)
print("ANALYSIS 1: PARETO FRONTIER")
print("="*100)
results_pareto = perform_regression_analysis(
    df_clean,
    use_pareto_only=True,
    min_date=datetime(2024, 4, 1)
)

# Run analysis on all models
print("\n" + "="*100)
print("ANALYSIS 2: ALL MODELS")
print("="*100)
results_all = perform_regression_analysis(
    df_clean,
    use_pareto_only=False,
    min_date=datetime(2024, 4, 1)
)

# Create comprehensive summary table
print("\n\n" + "="*120)
print("COMPREHENSIVE SUMMARY TABLE")
print("="*120 + "\n")

summary_rows = []

# Add Pareto frontier results
for _, row in results_pareto['results_df'].iterrows():
    summary_rows.append({
        'Sample': 'Pareto Frontier',
        'Model': row['Model'],
        'N': int(row['N_Models']),
        'Time Coef (logits/yr)': row['Time_Coefficient'],
        'Time Std Err': row['Time_Std_Err'],
        'Time p-value': row['Time_p_value'],
        '95% CI Lower': row['Time_CI_Lower'],
        '95% CI Upper': row['Time_CI_Upper'],
        'Annual Δ (% pts)': row['Annual_Improvement_PctPts'],
        'Price Coef': row['Price_Coefficient'],
        'Price p-value': row['Price_p_value'],
        'R²': row['R_Squared'],
        'Adj R²': row['Adj_R_Squared']
    })

# Add all models results
for _, row in results_all['results_df'].iterrows():
    summary_rows.append({
        'Sample': 'All Models',
        'Model': row['Model'],
        'N': int(row['N_Models']),
        'Time Coef (logits/yr)': row['Time_Coefficient'],
        'Time Std Err': row['Time_Std_Err'],
        'Time p-value': row['Time_p_value'],
        '95% CI Lower': row['Time_CI_Lower'],
        '95% CI Upper': row['Time_CI_Upper'],
        'Annual Δ (% pts)': row['Annual_Improvement_PctPts'],
        'Price Coef': row['Price_Coefficient'],
        'Price p-value': row['Price_p_value'],
        'R²': row['R_Squared'],
        'Adj R²': row['Adj_R_Squared']
    })

summary_table = pd.DataFrame(summary_rows)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.precision', 4)

print(summary_table.to_string(index=False))

# Save to CSV
summary_table.to_csv('results/gpqa_regression_summary.csv', index=False)
print("\n\nSummary table saved to: results/gpqa_regression_summary.csv")

# Print key insights
print("\n\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)

pareto_no_control = results_pareto['results_df'][results_pareto['results_df']['Model'] == 'Without Price Control'].iloc[0]
pareto_with_control = results_pareto['results_df'][results_pareto['results_df']['Model'] == 'With Price Control'].iloc[0]

print("\n1. PARETO FRONTIER (Best Models Over Time):")
print(f"   N = {int(pareto_no_control['N_Models'])} models\n")

print("   Without controlling for price:")
print(f"   - Annual improvement: {pareto_no_control['Annual_Improvement_Logits']:.3f} logits/yr")
print(f"   - Approximate: {pareto_no_control['Annual_Improvement_PctPts']:.2f} percentage points/yr")
print(f"   - R² = {pareto_no_control['R_Squared']:.3f}")
print(f"   - p-value: {pareto_no_control['Time_p_value']:.4f}\n")

print("   Controlling for price:")
print(f"   - Annual improvement: {pareto_with_control['Annual_Improvement_Logits']:.3f} logits/yr")
print(f"   - Approximate: {pareto_with_control['Annual_Improvement_PctPts']:.2f} percentage points/yr")
print(f"   - Price effect: {pareto_with_control['Price_Coefficient']:.3f} logits per 10x price increase")
print(f"   - R² = {pareto_with_control['R_Squared']:.3f}")
print(f"   - Time p-value: {pareto_with_control['Time_p_value']:.4f}")
print(f"   - Price p-value: {pareto_with_control['Price_p_value']:.4f}\n")

coef_change = pareto_with_control['Annual_Improvement_Logits'] - pareto_no_control['Annual_Improvement_Logits']
pct_change = (coef_change / pareto_no_control['Annual_Improvement_Logits']) * 100

print("   Effect of controlling for price:")
print(f"   - Coefficient change: {coef_change:.3f} logits/yr ({pct_change:+.1f}%)")
if coef_change < 0:
    print("   - Interpretation: Some of the apparent progress is due to higher prices")
else:
    print("   - Interpretation: Progress is independent of (or enhanced beyond) price effects")

print("\n" + "="*100)
