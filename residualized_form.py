#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor, RidgeCV
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from scipy import stats
#%%

#read differnet df 

df = pd.read_csv('inference_data_new_large.csv')
print(df.columns)
#convert price to float
# df['Output Price\nUSD/1M Tokens'] = df['Output Price\nUSD/1M Tokens'].str.replace('$', '').astype(float)
# df['Lowest Blended Price AA'] = df['Lowest Blended Price AA'].astype(float)
# df['Blended\nUSD/1M Tokens'] = df['Blended\nUSD/1M Tokens'].str.replace('$', '').astype(float)


#convert release date to datetime where release date is not nan
df['Release Date'] = pd.to_datetime(df['Release Date'])



# # Create Active Parameters column by choosing Known Active Parameters when available, otherwise use Parameters
df['Active Parameters'] = np.where(
    df['Known Active Parameters'].notna(),
    df['Known Active Parameters'],
    df['Parameters']
)

model, data, results = plot_price_mmlu_regression(df, open_license_only=False, price_column="total price swe", 
exclude_dominated=False, benchmark_col="epoch_swe", min_mmlu=20, max_mmlu=80, exclude_reasoning=False, use_huber=False, pareto_frontier_only=True)




#%%
def _ols_residualize(x, z):
    """Return residuals of x ~ z (adds intercept)."""
    Z = np.column_stack([np.ones(len(z)), z])
    coef, *_ = np.linalg.lstsq(Z, x, rcond=None)
    return x - Z @ coef

def _vif_two_predictors(x1, x2):
    """VIFs when there are exactly two predictors."""
    r = np.corrcoef(x1, x2)[0,1]
    r2 = r**2
    vif = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
    return vif, r

def plot_price_mmlu_regression(
    df,
    open_license_only=False,
    min_mmlu=40,
    max_mmlu=70,
    price_column='Output Price\nUSD/1M Tokens',
    exclude_dominated=False,
    benchmark_col='MMLU-Pro (Reasoning & Knowledge)',
    exclude_reasoning=False,
    use_huber=False,
    huber_epsilon=1.35,
    huber_max_iter=100,
    pareto_frontier_only=False,
    # ==== NEW/CHANGED ====
    collinearity_strategy="residualize_benchmark",  # {"none","residualize_benchmark","residualize_time","ridge"}
    standardize=False,                               # center/scale features before fit (recommended for ridge)
    ridge_alphas=np.logspace(-6, 3, 30)              # grid for RidgeCV if used
):
    """
    Plot log(Price) = alpha*time + beta*Benchmark + c

    collinearity_strategy:
        - "none": original OLS/Huber on [time, benchmark]
        - "residualize_benchmark": regress Benchmark ~ Time, use residual(Benchmark) in main model
        - "residualize_time": regress Time ~ Benchmark, use residual(Time) in main model
        - "ridge": RidgeCV on [time, benchmark] (no CIs)
    """
    # Column names
    mmlu_col = benchmark_col
    price_col = price_column
    license_col = 'License'
    reasoning_col = 'Reasoning_TF'

    # Work on a copy
    df_work = df.copy()

    # 1) Convert Benchmark "XX%" → float when it looks like a percent
    df_work[mmlu_col] = (
        df_work[mmlu_col].astype(str)
                         .str.replace('%', '', regex=False)
                         .replace(['nan','None'], np.nan)
                         .astype(float)
    )

    # 2) Convert price "$X,XXX" → float
    df_work[price_col] = (
        df_work[price_col].astype(str)
                          .str.replace('[$,]', '', regex=True)
    )
    df_work[price_col] = pd.to_numeric(df_work[price_col], errors='coerce')

    # 3) Optionally filter to open-license only
    if open_license_only:
        df_work = df_work[
            df_work[license_col].notna() &
            df_work[license_col].str.contains('open', case=False, na=False)
        ]

    # 4) Optionally filter out reasoning models
    if exclude_reasoning and reasoning_col in df_work.columns:
        df_work = df_work[df_work[reasoning_col] != True]

    # 5) Filter to rows with all necessary data
    df_sub = df_work.dropna(subset=['Release Date', price_col, mmlu_col])
    df_sub = df_sub[(df_sub[price_col] > 0) & (df_sub[mmlu_col] > 0)]

    # 6) Filter by benchmark range
    df_sub = df_sub[(df_sub[mmlu_col] >= min_mmlu) & (df_sub[mmlu_col] <= max_mmlu)]

    # 7) Optionally filter out Pareto dominated models (for display only)
    df_sub_display = df_sub.copy()
    if exclude_dominated:
        df_sub_display = df_sub_display.sort_values('Release Date')
        non_dominated = []
        for i, row in df_sub_display.iterrows():
            dominated = False
            for j in non_dominated:
                prev_row = df_sub_display.loc[j]
                if (prev_row[mmlu_col] >= row[mmlu_col] and 
                    prev_row[price_col] <= row[price_col] and
                    (prev_row[mmlu_col] > row[mmlu_col] or prev_row[price_col] < row[price_col])):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(i)
                # purge those it dominates
                new_non_dominated = []
                for j in non_dominated[:-1]:
                    prev_row = df_sub_display.loc[j]
                    if not (row[mmlu_col] >= prev_row[mmlu_col] and 
                            row[price_col] <= prev_row[price_col] and
                            (row[mmlu_col] > prev_row[mmlu_col] or row[price_col] < row[price_col])):
                        new_non_dominated.append(j)
                non_dominated = new_non_dominated + [i]
        df_sub_display = df_sub_display.loc[non_dominated]

    # 8) Choose data for regression
    if pareto_frontier_only:
        df_regression = df_sub.sort_values('Release Date').copy()
        pareto_indices = []
        for date in df_regression['Release Date'].unique():
            available = df_regression[df_regression['Release Date'] <= date].copy()
            available = available.sort_values([price_col, mmlu_col])
            frontier = []
            for i, row in available.iterrows():
                dominated = False
                for j in frontier:
                    fr = available.loc[j]
                    if (fr[mmlu_col] >= row[mmlu_col] and 
                        fr[price_col] <= row[price_col] and
                        (fr[mmlu_col] > row[mmlu_col] or fr[price_col] < row[price_col])):
                        dominated = True
                        break
                if not dominated:
                    frontier.append(i)
                    new_frontier = []
                    for j in frontier[:-1]:
                        fr = available.loc[j]
                        if not (row[mmlu_col] >= fr[mmlu_col] and 
                                row[price_col] <= fr[price_col] and
                                (row[mmlu_col] > fr[mmlu_col] or row[price_col] < fr[price_col])):
                            new_frontier.append(j)
                    frontier = new_frontier + [i]
            # add models released exactly on this date that are on the frontier
            current_date_models = df_regression[df_regression['Release Date'] == date]
            for i, _ in current_date_models.iterrows():
                if i in frontier:
                    pareto_indices.append(i)
        df_regression = df_regression.loc[list(set(pareto_indices))]
    else:
        df_regression = df_sub.copy()

    if len(df_regression) < 3:
        print(f"Warning: Only {len(df_regression)} data points available for regression. Need at least 3.")
        return None, None, None

    # 9) Prepare variables
    df_regression = df_regression.sort_values('Release Date').copy()
    df_regression['Date_Ordinal'] = df_regression['Release Date'].map(datetime.toordinal)

    # Use a numerically stable time scale (years since min date)
    min_ord = df_regression['Date_Ordinal'].min()
    time_years = (df_regression['Date_Ordinal'].values - min_ord) / 365.0
    bench = df_regression[mmlu_col].values.astype(float)

    # Diagnostics: VIF & correlation (original variables)
    vif, corr = _vif_two_predictors(time_years, bench)

    # ==== NEW/CHANGED: build features per selected strategy ====
    if collinearity_strategy == "residualize_benchmark":
        bench_resid = _ols_residualize(bench, time_years)
        X = np.column_stack([time_years, bench_resid])
        feature_names = ["time_years", f"{mmlu_col}_resid(time)"]
        strategy_label = "Residualize benchmark ~ time"
        # Note: α matches OLS with both regressors; β is effect of benchmark net of time trend.
    elif collinearity_strategy == "residualize_time":
        time_resid = _ols_residualize(time_years, bench)
        X = np.column_stack([time_resid, bench])
        feature_names = ["time_years_resid(benchmark)", mmlu_col]
        strategy_label = "Residualize time ~ benchmark"
        # Note: β matches OLS with both regressors; α is time effect net of benchmark.
    elif collinearity_strategy == "ridge":
        X = np.column_stack([time_years, bench])
        feature_names = ["time_years", mmlu_col]
        strategy_label = "RidgeCV (L2)"
    else:  # "none"
        X = np.column_stack([time_years, bench])
        feature_names = ["time_years", mmlu_col]
        strategy_label = "Standard (no special treatment)"

    # Target: log(Price)
    y = np.log(df_regression[price_col].values)

    # Optional standardization (recommended for ridge)
    scaler = None
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
        X_for_fit = scaler.transform(X)
    else:
        X_for_fit = X

    # 10) Fit regression
    reg_type = "OLS"
    alpha_coef = beta_coef = c = np.nan
    factor_change_lower = factor_change_upper = None
    factor_decrease_lower = factor_decrease_upper = None

    if collinearity_strategy == "ridge":
        model = RidgeCV(alphas=ridge_alphas, fit_intercept=True).fit(X_for_fit, y)
        y_pred = model.predict(X_for_fit)
        r2 = model.score(X_for_fit, y)
        # Back out coefficients on original scale if standardized
        if standardize:
            # y = a + b*(Xstd), Xstd=(X-μ)/σ  => coefficients on raw X: b/σ, intercept: a - Σ (b*μ/σ)
            coefs_raw = model.coef_ / scaler.scale_
            intercept_raw = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
            alpha_coef, beta_coef = coefs_raw
            c = intercept_raw
        else:
            alpha_coef, beta_coef = model.coef_
            c = model.intercept_
        use_huber_fit = False
        reg_type = "RidgeCV"
    else:
        if use_huber:
            model = HuberRegressor(epsilon=huber_epsilon, max_iter=huber_max_iter).fit(X_for_fit, y)
            y_pred = model.predict(X_for_fit)
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            reg_type = "Huber"
        else:
            model = LinearRegression().fit(X_for_fit, y)
            y_pred = model.predict(X_for_fit)
            r2 = model.score(X_for_fit, y)
            reg_type = "OLS"

        # Map coefficients back if standardized
        if standardize:
            coefs_raw = model.coef_ / scaler.scale_
            intercept_raw = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
            alpha_coef, beta_coef = coefs_raw
            c = intercept_raw
        else:
            alpha_coef, beta_coef = model.coef_
            c = model.intercept_

    # 11) Annual change (α is per-year because time_years is in years)
    annual_log_change = alpha_coef
    annual_pct_change = (np.exp(annual_log_change) - 1) * 100
    factor_change_per_year = np.exp(annual_log_change)
    factor_decrease_per_year = 1 / factor_change_per_year if factor_change_per_year < 1 else None

    # 12) Confidence intervals (OLS only; skip for Huber/Ridge)
    if reg_type == "OLS":
        n = len(df_regression)
        p = X.shape[1]
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - p - 1)
        Xc = X_for_fit - (X_for_fit.mean(axis=0))
        cov_matrix = np.linalg.inv(Xc.T @ Xc) * mse
        # alpha is the first column coefficient in our constructed X_for_fit
        se_alpha_std = np.sqrt(cov_matrix[0, 0])
        # If standardized, convert SE to raw-scale for time coefficient
        if standardize:
            se_alpha = se_alpha_std / scaler.scale_[0]
        else:
            se_alpha = se_alpha_std
        t_stat = stats.t.ppf(0.95, n - p - 1)  # 90% CI
        annual_log_change_lower = annual_log_change - t_stat * se_alpha
        annual_log_change_upper = annual_log_change + t_stat * se_alpha
        factor_change_lower = np.exp(annual_log_change_lower)
        factor_change_upper = np.exp(annual_log_change_upper)
        if factor_change_per_year < 1:
            factor_decrease_lower = 1 / factor_change_upper
            factor_decrease_upper = 1 / factor_change_lower

    # 13) Predictions for plotting at median benchmark (same as before)
    min_ord_plot, max_ord_plot = df_regression['Date_Ordinal'].min(), df_regression['Date_Ordinal'].max()
    x_range_ord = np.linspace(min_ord_plot, max_ord_plot, 100)
    x_dates = [datetime.fromordinal(int(d)) for d in x_range_ord]
    x_years = (x_range_ord - min_ord) / 365.0
    median_bench = df_regression[mmlu_col].median()

    # Build prediction matrix to reflect chosen strategy
    if collinearity_strategy == "residualize_benchmark":
        bench_median_resid = _ols_residualize(
            np.full_like(x_years, median_bench, dtype=float),
            x_years
        )
        X_pred = np.column_stack([x_years, bench_median_resid])
    elif collinearity_strategy == "residualize_time":
        time_pred_resid = _ols_residualize(x_years, np.full_like(x_years, median_bench, dtype=float))
        X_pred = np.column_stack([time_pred_resid, np.full_like(x_years, median_bench, dtype=float)])
    else:
        X_pred = np.column_stack([x_years, np.full_like(x_years, median_bench, dtype=float)])

    if standardize:
        X_pred_fit = scaler.transform(X_pred)
    else:
        X_pred_fit = X_pred

    y_pred_plot = model.predict(X_pred_fit)

    # 14) Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        df_sub_display['Release Date'], 
        df_sub_display[price_col],
        c=df_sub_display[mmlu_col], 
        cmap='viridis', 
        alpha=0.7,
        s=60,
        label='Data points'
    )

    if pareto_frontier_only:
        plt.scatter(
            df_regression['Release Date'], 
            df_regression[price_col],
            facecolors='none',
            edgecolors='red',
            s=100,
            linewidth=2,
            label='Pareto frontier (used for regression)'
        )

    cbar = plt.colorbar(scatter)
    benchmark_name = benchmark_col.split(' (')[0]
    cbar.set_label(f'{benchmark_name} Score (%)')

    data_source = "Pareto frontier only" if pareto_frontier_only else "all data"
    if factor_decrease_per_year:
        regression_label = (f'{reg_type} ({strategy_label}, at median {benchmark_name}={median_bench:.1f}%)\n'
                            f'Annual change: {annual_pct_change:.2f}%/yr\n'
                            f'Factor decrease: {factor_decrease_per_year:.3f}×/yr'
                            + (f' (90% CI: [{factor_decrease_lower:.3f}, {factor_decrease_upper:.3f}])' if factor_decrease_lower is not None else '') +
                            f'\nR² = {r2:.3f}')
    else:
        regression_label = (f'{reg_type} ({strategy_label}, at median {benchmark_name}={median_bench:.1f}%)\n'
                            f'Annual change: {annual_pct_change:.2f}%/yr\n'
                            f'Factor change: {factor_change_per_year:.3f}×/yr'
                            + (f' (90% CI: [{factor_change_lower:.3f}, {factor_change_upper:.3f}])' if factor_change_lower is not None else '') +
                            f'\nR² = {r2:.3f}')

    plt.plot(x_dates, np.exp(y_pred_plot), 'r-', lw=3, label=regression_label)

    plt.yscale('log')
    plt.xlabel('Release Date')
    plt.ylabel('Price (USD per 1M tokens)')

    lic_label = 'open-license only' if open_license_only else 'all licenses'
    mmlu_range = f"{benchmark_name} ∈ [{min_mmlu},{max_mmlu}]%"
    price_type = price_col.replace('\n', ' ')
    pareto_label = "non-dominated models only" if exclude_dominated else "all models"
    reasoning_label = "excluding reasoning models" if exclude_reasoning else "including reasoning models"
    frontier_label = "Pareto frontier regression" if pareto_frontier_only else "standard regression"
    
    plt.title(f'Price vs Time & {benchmark_name} {reg_type} Regression ({lic_label}, {mmlu_range}, {price_type}, {pareto_label}, {reasoning_label}, {frontier_label})\n'
              f'[{strategy_label}]  log(Price) = {alpha_coef:.6f}×time + {beta_coef:.3f}×{benchmark_name} + {c:.3f}')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Print results + diagnostics
    print(f"\nRegression Results ({reg_type}, {strategy_label}):")
    print(f"Data used: {data_source}")
    print(f"Model (years since first release): log(Price) = {alpha_coef:.6f}×time + {beta_coef:.3f}×{benchmark_name} + {c:.3f}")
    print(f"R² score: {r2:.4f}")
    print(f"\nDiagnostics (original features): corr(time, {benchmark_name}) = {corr:.3f}, VIF ≈ {vif:.2f}")
    print(f"Annual percentage change: {annual_pct_change:.2f}%/yr")
    if factor_decrease_per_year:
        print(f"Annual factor decrease: {factor_decrease_per_year:.3f}×/yr")
        if factor_decrease_lower is not None:
            print(f"90% CI for factor decrease: [{factor_decrease_lower:.3f}, {factor_decrease_upper:.3f}]")
    else:
        print(f"Annual factor change: {factor_change_per_year:.3f}×/yr")
        if factor_change_lower is not None:
            print(f"90% CI for factor change: [{factor_change_lower:.3f}, {factor_change_upper:.3f}]")
    print(f"{benchmark_name} coefficient (beta): {beta_coef:.3f}")
    print(f"Intercept (c): {c:.3f}")
    print(f"\nData points used for regression: {len(df_regression)}")
    print(f"Data points displayed: {len(df_sub_display)}")

    # Return everything, including diagnostics
    return model, df_regression, {
        'alpha': alpha_coef,
        'beta': beta_coef, 
        'c': c,
        'annual_pct_change': annual_pct_change,
        'factor_change_per_year': factor_change_per_year,
        'factor_decrease_per_year': factor_decrease_per_year if factor_change_per_year < 1 else None,
        'factor_change_ci_lower': factor_change_lower,
        'factor_change_ci_upper': factor_change_upper,
        'factor_decrease_ci_lower': factor_decrease_lower if factor_change_per_year < 1 else None,
        'factor_decrease_ci_upper': factor_decrease_upper if factor_change_per_year < 1 else None,
        'r2_score': r2,
        'regression_type': reg_type,
        'pareto_frontier_only': pareto_frontier_only,
        # ==== NEW ====
        'collinearity_strategy': collinearity_strategy,
        'vif_time_benchmark': vif,
        'corr_time_benchmark': corr
    }
#%%

model, data, results = plot_price_mmlu_regression(
    df,
    open_license_only=False,
    price_column="total price swe",         # or your preferred price column
    exclude_dominated=False,
    benchmark_col="epoch_swe",                # or "SWE-score" if that’s your column
    min_mmlu=0, max_mmlu=100,
    exclude_reasoning=False,
    use_huber=False,
    pareto_frontier_only=True,
    collinearity_strategy="ridge",  # <<< key change
    standardize=False
)

# %%
