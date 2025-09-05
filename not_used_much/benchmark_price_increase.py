

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, QuantileRegressor
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

#%%

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
print(df.columns)
#%%
def plot_benchmark_price_vs_time(
    csv_file="price_reduction_models.csv",
    price_col="Benchmark Cost USD",
    benchmark_col="epoch_gpqa", 
    min_benchmark=None,
    max_benchmark=None,
    open_license_only=False,
    include_chinese=None,
    include_reasoning_models=True,
    min_date=None,
    show_model_names=False,
    confidence_interval=True,
    fit_overall_trend=True,
    exclude_dominated=False,
    use_quantile_regression=False,
    quantile=0.5,
    record_price_trend=False,
    figsize=(14, 8)
):
    """
    Graph total price for any benchmark vs release date with overall fit capability.
    
    Parameters:
    - csv_file: Path to CSV file (default: "price_reduction_models.csv")
    - price_col: Price column name (default: "Benchmark Cost USD")
    - benchmark_col: Benchmark column name (default: "epoch_gpqa")
    - min_benchmark: Minimum benchmark score to include
    - max_benchmark: Maximum benchmark score to include  
    - open_license_only: If True, only include open license models
    - include_chinese: Filter for Chinese models (True/False/None)
    - include_reasoning_models: Include reasoning models (default: True)
    - min_date: Minimum date to include (datetime or string)
    - show_model_names: Show model names on points
    - confidence_interval: Show 90% confidence intervals
    - fit_overall_trend: Fit overall trend line to all data
    - exclude_dominated: Exclude Pareto dominated models
    - use_quantile_regression: Use quantile regression instead of OLS (default: False)
    - quantile: Quantile for quantile regression (default: 0.5 for median)
    - record_price_trend: If True, fit additional trend to record (maximum) prices over time
    - figsize: Figure size tuple
    
    Returns:
    - model: Fitted regression model (record trend if record_price_trend=True, else overall trend)
    - df_filtered: Filtered dataframe used for analysis
    - stats_dict: Dictionary with regression statistics
    """
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} models from {csv_file}")
    
    # Work on a copy
    df_work = df.copy()
    
    # 1) Clean benchmark column
    if benchmark_col in df_work.columns:
        # Handle percentage format if present
        if df_work[benchmark_col].dtype == 'object':
            df_work[benchmark_col] = (
                df_work[benchmark_col].astype(str)
                                     .str.replace('%', '', regex=False)
                                     .str.replace('nan', '')
            )
        df_work[benchmark_col] = pd.to_numeric(df_work[benchmark_col], errors='coerce')
    else:
        print(f"Warning: Benchmark column '{benchmark_col}' not found!")
        print(f"Available columns: {list(df_work.columns)}")
        return None, None, None
    
    # 2) Clean price column
    if price_col in df_work.columns:
        if df_work[price_col].dtype == 'object':
            df_work[price_col] = (
                df_work[price_col].astype(str)
                                 .str.replace('[$,]', '', regex=True)
            )
        df_work[price_col] = pd.to_numeric(df_work[price_col], errors='coerce')
    else:
        print(f"Warning: Price column '{price_col}' not found!")
        print(f"Available columns: {list(df_work.columns)}")
        return None, None, None
    
    # 3) Handle Release Date
    if 'Release Date' in df_work.columns:
        df_work['Release Date'] = pd.to_datetime(df_work['Release Date'], errors='coerce')
    else:
        print("Warning: 'Release Date' column not found!")
        return None, None, None
    
    # 4) Apply filters
    # Filter out missing data
    df_sub = df_work.dropna(subset=['Release Date', price_col, benchmark_col])
    df_sub = df_sub[(df_sub[price_col] > 0) & (df_sub[benchmark_col] > 0)]
    
    # Open license filter
    if open_license_only and 'License' in df_sub.columns:
        df_sub = df_sub[
            df_sub['License'].notna() &
            df_sub['License'].str.contains('open', case=False, na=False)
        ]
    
    # Chinese models filter
    if include_chinese is not None and 'Chinese' in df_sub.columns:
        if include_chinese:
            df_sub = df_sub[df_sub['Chinese'] == True]
        else:
            df_sub = df_sub[(df_sub['Chinese'] != True) | (df_sub['Chinese'].isna())]
    
    # Reasoning models filter
    if not include_reasoning_models and 'Reasoning_TF' in df_sub.columns:
        df_sub = df_sub[(df_sub['Reasoning_TF'] != True) | (df_sub['Reasoning_TF'].isna())]
    
    # Date filter
    if min_date is not None:
        if isinstance(min_date, str):
            min_date = pd.to_datetime(min_date)
        df_sub = df_sub[df_sub['Release Date'] >= min_date]
    
    # Benchmark range filter
    if min_benchmark is not None:
        df_sub = df_sub[df_sub[benchmark_col] >= min_benchmark]
    if max_benchmark is not None:
        df_sub = df_sub[df_sub[benchmark_col] <= max_benchmark]
    
    # Exclude dominated models if requested
    if exclude_dominated:
        df_sub = df_sub.sort_values('Release Date')
        non_dominated = []
        
        for i, row in df_sub.iterrows():
            dominated = False
            for j in non_dominated:
                prev_row = df_sub.loc[j]
                # Dominated if previous model has better/equal benchmark AND lower/equal price
                if (prev_row[benchmark_col] >= row[benchmark_col] and 
                    prev_row[price_col] <= row[price_col] and
                    (prev_row[benchmark_col] > row[benchmark_col] or prev_row[price_col] < row[price_col])):
                    dominated = True
                    break
            
            if not dominated:
                non_dominated.append(i)
        
        df_sub = df_sub.loc[non_dominated]
    
    print(f"After filtering: {len(df_sub)} models")
    
    if len(df_sub) == 0:
        print("No data points remain after filtering!")
        return None, None, None
    
    # 5) Sort by date and prepare for analysis
    df_sub = df_sub.sort_values('Release Date')
    df_sub['Date_Ordinal'] = df_sub['Release Date'].map(datetime.toordinal)
    
    # 6) Fit overall trend if requested
    overall_model = None
    overall_stats = None
    if fit_overall_trend and len(df_sub) >= 3:
        # Prepare data for regression
        X = df_sub['Date_Ordinal'].values.reshape(-1, 1)
        y_log = np.log(df_sub[price_col].values)
        
        # Choose regression model based on option
        if use_quantile_regression:
            # Use quantile regression
            overall_model = QuantileRegressor(quantile=quantile, alpha=0, solver='highs').fit(X, y_log)
            # For quantile regression, we can't calculate R² in the same way
            # Instead we can compute pseudo R² based on quantile loss
            y_pred = overall_model.predict(X)
            # Calculate quantile loss for baseline (median of y) and model
            baseline_pred = np.full_like(y_log, np.quantile(y_log, quantile))
            
            def quantile_loss(y_true, y_pred, q):
                errors = y_true - y_pred
                return np.mean(np.maximum(q * errors, (q - 1) * errors))
            
            baseline_loss = quantile_loss(y_log, baseline_pred, quantile)
            model_loss = quantile_loss(y_log, y_pred, quantile)
            overall_r2 = 1 - (model_loss / baseline_loss) if baseline_loss > 0 else 0
        else:
            # Use standard linear regression
            overall_model = LinearRegression().fit(X, y_log)
            # Calculate R²
            overall_r2 = overall_model.score(X, y_log)
        
        # Calculate annual change
        overall_slope = overall_model.coef_[0] if hasattr(overall_model, 'coef_') else overall_model.coef_
        overall_annual_log_change = overall_slope * 365
        overall_annual_pct_change = (np.exp(overall_annual_log_change) - 1) * 100
        overall_factor_change_per_year = np.exp(overall_annual_log_change)
        
        # Express as factor decrease if decreasing
        if overall_factor_change_per_year < 1:
            overall_factor_decrease_per_year = 1 / overall_factor_change_per_year
        else:
            overall_factor_decrease_per_year = None
        
        # Generate prediction line
        min_ord, max_ord = df_sub['Date_Ordinal'].min(), df_sub['Date_Ordinal'].max()
        x_range = np.arange(min_ord, max_ord + 1)
        x_dates = [datetime.fromordinal(int(d)) for d in x_range]
        overall_y_pred_log = overall_model.predict(x_range.reshape(-1, 1))
        
        # Calculate confidence intervals if requested (only for OLS)
        if confidence_interval and len(df_sub) > 2 and not use_quantile_regression:
            n = len(df_sub)
            y_pred_sample = overall_model.predict(X)
            residuals = y_log - y_pred_sample
            mse = np.sum(residuals**2) / (n - 2)
            se = np.sqrt(mse / np.sum((X.flatten() - np.mean(X))**2))
            
            # t-value for 90% confidence interval
            t_val = stats.t.ppf(0.95, n - 2)
            
            # Confidence interval for slope
            ci_lower = overall_slope - t_val * se
            ci_upper = overall_slope + t_val * se
            
            # Plot confidence interval
            overall_y_lower = ci_lower * x_range + overall_model.intercept_
            overall_y_upper = ci_upper * x_range + overall_model.intercept_
            
            # CI for annual factors
            annual_factor_lower = 1 / np.exp(ci_upper * 365) if overall_factor_decrease_per_year else np.exp(ci_lower * 365)
            annual_factor_upper = 1 / np.exp(ci_lower * 365) if overall_factor_decrease_per_year else np.exp(ci_upper * 365)
            overall_ci_text = f" (90% CI: {annual_factor_lower:.1f}x-{annual_factor_upper:.1f}x)"
        else:
            overall_ci_text = ""
        
        # Store overall trend line data
        overall_trend_x_dates = x_dates
        overall_trend_y_pred = np.exp(overall_y_pred_log)
        
        if confidence_interval and len(df_sub) > 2 and not use_quantile_regression:
            overall_ci_x_dates = x_dates
            overall_ci_y_lower = np.exp(overall_y_lower)
            overall_ci_y_upper = np.exp(overall_y_upper)
        else:
            overall_ci_x_dates = None
            overall_ci_y_lower = None
            overall_ci_y_upper = None
        
        # Create overall trend label
        regression_type = f"Q{int(quantile*100)} Quantile" if use_quantile_regression else "OLS"
        # Remove R² from legend label
        if overall_factor_decrease_per_year:
            overall_trend_label = f'Overall {regression_type} Trend: {overall_factor_decrease_per_year:.1f}x cheaper/yr{overall_ci_text}'
        else:
            overall_trend_label = f'Overall {regression_type} Trend: {overall_factor_change_per_year:.1f}x/yr{overall_ci_text}'
        
        # Store overall statistics
        overall_stats = {
            'slope': overall_slope,
            'annual_pct_change': overall_annual_pct_change,
            'factor_change_per_year': overall_factor_change_per_year,
            'factor_decrease_per_year': overall_factor_decrease_per_year,
            'r2': overall_r2,
            'n_points': len(df_sub),
            'regression_type': regression_type,
            'quantile': quantile if use_quantile_regression else None,
            'trend_description': 'Overall'
        }
    
    # 7) Fit record price trend if requested
    record_model = None
    record_stats = None
    if record_price_trend and len(df_sub) >= 3:
        # Create record price trend: for each date, take the maximum price seen up to that date
        df_record = df_sub.copy()
        df_record = df_record.sort_values('Release Date')
        df_record['Cumulative_Max_Price'] = df_record[price_col].cummax()
        
        # Only keep records where the price equals the cumulative maximum (actual record setters)
        df_regression = df_record[df_record[price_col] == df_record['Cumulative_Max_Price']].copy()
        
        # Remove duplicate dates, keeping the last record for each date
        df_regression = df_regression.drop_duplicates(subset=['Release Date'], keep='last')
        
        print(f"Record price trend: Using {len(df_regression)} record-setting models")
        
        if len(df_regression) >= 3:
            # Prepare data for regression
            X_record = df_regression['Date_Ordinal'].values.reshape(-1, 1)
            y_log_record = np.log(df_regression[price_col].values)
            
            # Choose regression model based on option
            if use_quantile_regression:
                # Use quantile regression
                record_model = QuantileRegressor(quantile=quantile, alpha=0, solver='highs').fit(X_record, y_log_record)
                # For quantile regression, we can't calculate R² in the same way
                # Instead we can compute pseudo R² based on quantile loss
                y_pred_record = record_model.predict(X_record)
                # Calculate quantile loss for baseline (median of y) and model
                baseline_pred_record = np.full_like(y_log_record, np.quantile(y_log_record, quantile))
                
                def quantile_loss(y_true, y_pred, q):
                    errors = y_true - y_pred
                    return np.mean(np.maximum(q * errors, (q - 1) * errors))
                
                baseline_loss_record = quantile_loss(y_log_record, baseline_pred_record, quantile)
                model_loss_record = quantile_loss(y_log_record, y_pred_record, quantile)
                record_r2 = 1 - (model_loss_record / baseline_loss_record) if baseline_loss_record > 0 else 0
            else:
                # Use standard linear regression
                record_model = LinearRegression().fit(X_record, y_log_record)
                # Calculate R²
                record_r2 = record_model.score(X_record, y_log_record)
            
            # Calculate annual change
            record_slope = record_model.coef_[0] if hasattr(record_model, 'coef_') else record_model.coef_
            record_annual_log_change = record_slope * 365
            record_annual_pct_change = (np.exp(record_annual_log_change) - 1) * 100
            record_factor_change_per_year = np.exp(record_annual_log_change)
            
            # Express as factor decrease if decreasing
            if record_factor_change_per_year < 1:
                record_factor_decrease_per_year = 1 / record_factor_change_per_year
            else:
                record_factor_decrease_per_year = None
            
            # Generate prediction line
            record_y_pred_log = record_model.predict(x_range.reshape(-1, 1))
            
            # Store record trend line data
            record_trend_x_dates = x_dates
            record_trend_y_pred = np.exp(record_y_pred_log)
            
            # Create record trend label
            # Remove R² from legend label
            if record_factor_decrease_per_year:
                record_trend_label = f'Record Price {regression_type} Trend: {record_factor_decrease_per_year:.1f}x cheaper/yr'
            else:
                record_trend_label = f'Record Price {regression_type} Trend: {record_factor_change_per_year:.1f}x/yr'
            
            # Store record statistics
            record_stats = {
                'slope': record_slope,
                'annual_pct_change': record_annual_pct_change,
                'factor_change_per_year': record_factor_change_per_year,
                'factor_decrease_per_year': record_factor_decrease_per_year,
                'r2': record_r2,
                'n_points': len(df_regression),
                'regression_type': regression_type,
                'quantile': quantile if use_quantile_regression else None,
                'trend_description': 'Record Price'
            }
        else:
            record_trend_x_dates = None
            record_trend_y_pred = None
            record_trend_label = None
            print("Not enough record price points for trend fitting")
    else:
        df_regression = None
        record_trend_x_dates = None
        record_trend_y_pred = None
        record_trend_label = None
    
    # 8) Generate default titles and labels for customization
    benchmark_name = benchmark_col.replace('_', ' ').title()
    
    # Create title
    filter_parts = []
    if open_license_only:
        filter_parts.append("open-license only")
    if include_chinese is True:
        filter_parts.append("Chinese models")
    elif include_chinese is False:
        filter_parts.append("non-Chinese models")
    if not include_reasoning_models:
        filter_parts.append("non-reasoning models")
    if exclude_dominated:
        filter_parts.append("non-dominated only")
    
    filter_text = f" ({', '.join(filter_parts)})" if filter_parts else ""
    
    benchmark_range = ""
    if min_benchmark is not None or max_benchmark is not None:
        min_val = min_benchmark if min_benchmark is not None else "min"
        max_val = max_benchmark if max_benchmark is not None else "max"
        benchmark_range = f" | {benchmark_name}: [{min_val}, {max_val}]"
    
    # Default text values - these will be used for setting labels
    default_title_text = f'{price_col.replace("_", " ")} vs Release Date{filter_text}{benchmark_range}'
    default_xlabel_text = 'Date'
    default_ylabel_text = f'{price_col.replace("_", " ")} (USD)'
    default_colorbar_label = f'{benchmark_name} Score'
    
    ####################################################################################
    # GRAPH APPEARANCE SETTINGS - EDIT THIS SECTION TO CUSTOMIZE GRAPH LOOK
    ####################################################################################
    
    # Figure setup
    plt.figure(figsize=figsize)
    
    # Scatter plot settings
    scatter_size = 120  # Increased from 80
    scatter_alpha = 0.7
    scatter_cmap = 'viridis'
    scatter_edge_color = 'white'
    scatter_edge_width = 1.0  # Increased from 0.5
    
    # Create scatter plot
    scatter = plt.scatter(
        df_sub['Release Date'], 
        df_sub[price_col],
        c=df_sub[benchmark_col], 
        cmap=scatter_cmap, 
        alpha=scatter_alpha,
        s=scatter_size,
        edgecolors=scatter_edge_color,
        linewidth=scatter_edge_width
    )
    
    # Highlight record price points if using record price trend
    if record_price_trend and df_regression is not None:
        record_highlight_size = 180  # Increased from 120
        record_highlight_color = 'red'
        record_highlight_alpha = 0.8
        record_highlight_marker = 's'  # square marker
        
        plt.scatter(
            df_regression['Release Date'],
            df_regression[price_col],
            s=record_highlight_size,
            color=record_highlight_color,
            alpha=record_highlight_alpha,
            marker=record_highlight_marker,
            facecolors='none',
            edgecolors=record_highlight_color,
            linewidth=3,  # Increased from 2
            label='Record Price Models'
        )
    
    # Colorbar settings - CUSTOMIZABLE SECTION
    colorbar_label = default_colorbar_label  # CUSTOMIZABLE - Change this line to set custom colorbar label
    colorbar_label = "GPQA-Diamond Score"  # Custom colorbar label
    colorbar_fontsize = 22  # Increased from 16
    colorbar_tick_labelsize = 20  # Increased from 14
    colorbar_shrink = 1.0  # Controls colorbar height relative to plot
    colorbar_aspect = 20   # Controls colorbar width (higher = thinner)
    colorbar_pad = 0.02    # Distance from plot to colorbar
    
    cbar = plt.colorbar(
        scatter, 
        shrink=colorbar_shrink, 
        aspect=colorbar_aspect, 
        pad=colorbar_pad
    )
    cbar.set_label(colorbar_label, fontsize=colorbar_fontsize, fontweight='bold')
    cbar.ax.tick_params(labelsize=colorbar_tick_labelsize)
    
    # Confidence interval settings (if applicable)
    if overall_ci_x_dates is not None and overall_ci_y_lower is not None:
        ci_alpha = 0.2
        ci_color = 'blue'
        ci_label = '90% Confidence Interval (Overall)'
        
        plt.fill_between(
            overall_ci_x_dates,
            overall_ci_y_lower,
            overall_ci_y_upper,
            alpha=ci_alpha,
            color=ci_color,
            label=ci_label
        )
    
    # Overall trend line settings (if applicable)
    if overall_trend_x_dates is not None and overall_trend_y_pred is not None:
        overall_trend_color = 'blue'
        overall_trend_linewidth = 4  # Increased from 3
        overall_trend_alpha = 0.8
        
        plt.plot(
            overall_trend_x_dates, 
            overall_trend_y_pred, 
            color=overall_trend_color, 
            linewidth=overall_trend_linewidth,
            alpha=overall_trend_alpha,
            label=overall_trend_label
        )
    
    # Record trend line settings (if applicable)
    if record_trend_x_dates is not None and record_trend_y_pred is not None:
        record_trend_color = 'red'
        record_trend_linewidth = 4  # Increased from 3
        record_trend_alpha = 0.8
        record_trend_linestyle = '--'  # Dashed line to distinguish from overall trend
        
        plt.plot(
            record_trend_x_dates, 
            record_trend_y_pred, 
            color=record_trend_color, 
            linewidth=record_trend_linewidth,
            alpha=record_trend_alpha,
            linestyle=record_trend_linestyle,
            label=record_trend_label
        )
    
    # Model name annotations (if requested)
    if show_model_names and 'Model' in df_sub.columns:
        annotation_fontsize = 12  # Increased from 8
        annotation_alpha = 0.8
        annotation_offset_x = 5
        annotation_offset_y = 5
        
        for idx, row in df_sub.iterrows():
            plt.annotate(
                row['Model'], 
                (row['Release Date'], row[price_col]),
                xytext=(annotation_offset_x, annotation_offset_y), 
                textcoords='offset points',
                fontsize=annotation_fontsize,
                alpha=annotation_alpha
            )
    
    # Y-axis settings
    plt.yscale('log')
    
    # Axis labels - CUSTOMIZABLE SECTION
    xlabel_text = default_xlabel_text  # CUSTOMIZABLE - Change this line to set custom x-axis label
    xlabel_text = "Date"
    xlabel_fontsize = 26  # Increased from 20
    xlabel_fontweight = 'bold'
    
    ylabel_text = default_ylabel_text  # CUSTOMIZABLE - Change this line to set custom y-axis label
    ylabel_text = "Benchmark Price (GPQA-Diamond)"
    ylabel_fontsize = 26  # Increased from 20
    ylabel_fontweight = 'bold'
    
    plt.xlabel(xlabel_text, fontsize=xlabel_fontsize, fontweight=xlabel_fontweight)
    plt.ylabel(ylabel_text, fontsize=ylabel_fontsize, fontweight=ylabel_fontweight)
    
    # Title settings - CUSTOMIZABLE SECTION
    title_text = default_title_text  # CUSTOMIZABLE - Change this line to set custom title

    title_text = "Benchmark Price (GPQA-Diamond) vs Date"
    title_fontsize = 28  # Increased from 20
    title_fontweight = 'bold'
    title_pad = 20
    
    plt.title(title_text, fontsize=title_fontsize, fontweight=title_fontweight, pad=title_pad)
    
    # X-axis tick settings (date labels) - Set ticks every 4 months
    tick_fontsize = 22  # Increased from 15
    
    # Set x-axis ticks to show every 4 months
    from matplotlib.dates import MonthLocator, DateFormatter
    ax = plt.gca()
    ax.xaxis.set_major_locator(MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    
    plt.tick_params(axis='x', labelsize=tick_fontsize, rotation=45)
    plt.tick_params(axis='y', labelsize=tick_fontsize)
    
    # Grid settings
    major_grid_linestyle = '--'
    major_grid_linewidth = 1.0  # Increased from 0.5
    major_grid_alpha = 0.7
    
    minor_grid_linestyle = ':'
    minor_grid_linewidth = 0.8  # Increased from 0.5
    minor_grid_alpha = 0.4
    
    plt.grid(True, which='major', linestyle=major_grid_linestyle, linewidth=major_grid_linewidth, alpha=major_grid_alpha)
    plt.grid(True, which='minor', linestyle=minor_grid_linestyle, linewidth=minor_grid_linewidth, alpha=minor_grid_alpha)
    
    # Legend settings
    if fit_overall_trend or record_price_trend:
        legend_fontsize = 19  # Increased from 18
        legend_location = 'upper left'
        legend_framealpha = 0.9
        
        plt.legend(fontsize=legend_fontsize, loc=legend_location, framealpha=legend_framealpha)
    
    # Layout
    plt.tight_layout()
    ####################################################################################
    # END GRAPH APPEARANCE SETTINGS
    ####################################################################################
    
    plt.show()
    
    # 9) Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Models analyzed: {len(df_sub)}")
    print(f"Price range: ${df_sub[price_col].min():.2f} - ${df_sub[price_col].max():.2f}")
    print(f"{benchmark_name} range: {df_sub[benchmark_col].min():.1f} - {df_sub[benchmark_col].max():.1f}")
    print(f"Date range: {df_sub['Release Date'].min().strftime('%Y-%m-%d')} to {df_sub['Release Date'].max().strftime('%Y-%m-%d')}")
    
    if overall_stats:
        print(f"\nOverall Trend Analysis ({overall_stats['regression_type']}):")
        if overall_stats['quantile']:
            print(f"Quantile: {overall_stats['quantile']}")
        print(f"Annual price change: {overall_stats['annual_pct_change']:.1f}%")
        if overall_stats['factor_decrease_per_year']:
            print(f"Price becomes {overall_stats['factor_decrease_per_year']:.1f}x cheaper each year")
        else:
            print(f"Price changes by factor of {overall_stats['factor_change_per_year']:.1f}x each year")
        metric_name = "Pseudo-R²" if use_quantile_regression else "R²"
        print(f"{metric_name} (goodness of fit): {overall_stats['r2']:.3f}")
    
    if record_stats:
        print(f"\nRecord Price Trend Analysis ({record_stats['regression_type']}):")
        print(f"Record price models used for trend: {record_stats['n_points']}")
        if record_stats['quantile']:
            print(f"Quantile: {record_stats['quantile']}")
        print(f"Annual price change: {record_stats['annual_pct_change']:.1f}%")
        if record_stats['factor_decrease_per_year']:
            print(f"Price becomes {record_stats['factor_decrease_per_year']:.1f}x cheaper each year")
        else:
            print(f"Price changes by factor of {record_stats['factor_change_per_year']:.1f}x each year")
        metric_name = "Pseudo-R²" if use_quantile_regression else "R²"
        print(f"{metric_name} (goodness of fit): {record_stats['r2']:.3f}")
    
    # Return the record model if available, otherwise the overall model
    # Combined stats dict with both trends
    combined_stats = {}
    if overall_stats:
        combined_stats['overall'] = overall_stats
    if record_stats:
        combined_stats['record'] = record_stats
    
    primary_model = record_model if record_model is not None else overall_model
    
    return primary_model, df_sub, combined_stats
#%%
# Example usage for GPQA:
plot_benchmark_price_vs_time(
    csv_file="inference_data_new_large.csv",
    price_col="total price swe",
    benchmark_col="epoch_swe", 
    open_license_only=False,
    min_date="2024-01-01",
    confidence_interval=False,
    fit_overall_trend=True,
    show_model_names=False, 
    use_quantile_regression=False,
    quantile=0.9,
    record_price_trend=True
)
# %%
