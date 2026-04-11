"""
Price/M tokens over time, stratified by parameter count order of magnitude.
Only includes models that are top on at least one benchmark in their parameter class and time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_csv('data/merged_with_training_compute.csv')

# ── Clean columns ──────────────────────────────────────────────────────

# Parameters (billions)
df['params_B'] = pd.to_numeric(df['Parameters'], errors='coerce')

# Release date
def parse_date(d):
    if pd.isna(d):
        return pd.NaT
    d = str(d).strip()
    for fmt in ['%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d']:
        try:
            return pd.to_datetime(d, format=fmt)
        except:
            continue
    try:
        return pd.to_datetime(d)
    except:
        return pd.NaT

df['date'] = df['Release Date'].apply(parse_date)

# Blended price USD/1M tokens
def parse_price(v):
    if pd.isna(v):
        return np.nan
    v = str(v).strip().replace('$', '').replace(',', '').strip()
    try:
        return float(v)
    except:
        return np.nan

df['blended_price'] = df['Blended\r\n USD/1M Tokens'].apply(parse_price)

# Also compute blended from input/output if blended is missing
def parse_price_col(v):
    if pd.isna(v):
        return np.nan
    v = str(v).strip().replace('$', '').replace(',', '').strip()
    try:
        return float(v)
    except:
        return np.nan

df['input_price'] = df['Input Price\r\n USD/1M Tokens'].apply(parse_price_col)
df['output_price'] = df['Output Price\r\n USD/1M Tokens'].apply(parse_price_col)

# Use blended if available, else compute 3:1 blend (3 input : 1 output)
df['price'] = df['blended_price']
mask_no_blended = df['price'].isna() & df['input_price'].notna() & df['output_price'].notna()
df.loc[mask_no_blended, 'price'] = (3 * df.loc[mask_no_blended, 'input_price'] + df.loc[mask_no_blended, 'output_price']) / 4

# ── Parse benchmark scores ────────────────────────────────────────────
benchmark_cols = {
    'MMLU-Pro': 'MMLU-Pro (Reasoning & Knowledge)',
    'GPQA-D': 'GPQA Diamond (Scientific Reasoning)',
    'HLE': "Humanity's Last Exam (Reasoning & Knowledge)",
    'LiveCode': 'LiveCodeBench (Coding)',
    'SciCode': 'SciCode (Coding)',
    'HumanEval': 'HumanEval (Coding)',
    'MATH-500': 'MATH-500 (Quantitative Reasoning)',
    'AIME': 'AIME 2024 (Competition Math)',
}

def parse_pct(v):
    if pd.isna(v):
        return np.nan
    v = str(v).strip().replace('%', '')
    try:
        val = float(v)
        # If value > 100, it was stored as e.g. 8700% meaning 87.00%
        # Actually looking at data: "87.10%" -> 87.1, "6900.00%" -> 6900
        # The 6900% ones look like AA Intelligence Index, not individual benchmarks
        # Individual benchmarks look like "87.10%" -> 87.1
        if val > 100:
            return val / 100.0  # e.g. 6900% -> 69 (but this shouldn't happen for these cols)
        return val
    except:
        return np.nan

for short_name, col_name in benchmark_cols.items():
    df[f'bench_{short_name}'] = df[col_name].apply(parse_pct)

bench_score_cols = [f'bench_{k}' for k in benchmark_cols.keys()]

# ── Assign parameter class ────────────────────────────────────────────
def param_class(p):
    if pd.isna(p):
        return None
    if p < 10:
        return '<10B'
    elif p < 100:
        return '10-100B'
    else:
        return '100B+'

df['param_class'] = df['params_B'].apply(param_class)

# ── Filter to models with params, date, price ─────────────────────────
valid = df.dropna(subset=['params_B', 'date', 'price']).copy()
valid = valid[valid['price'] > 0]  # Exclude free models (price=0 distorts analysis)
print(f"Models with params + date + price > 0: {len(valid)}")
print(f"Parameter class distribution:\n{valid['param_class'].value_counts().sort_index()}")

# ── Identify "top" models ─────────────────────────────────────────────
# A model is "top" if it achieves the best score on at least one benchmark
# among models in the same parameter class released on or before its release date.

def is_top_model(row, all_models):
    """Check if this model is top on any benchmark in its param class at its release time."""
    same_class = all_models[
        (all_models['param_class'] == row['param_class']) &
        (all_models['date'] <= row['date'])
    ]
    if len(same_class) <= 1:
        return True  # Only model in class at this time -> top by default

    for col in bench_score_cols:
        score = row[col]
        if pd.isna(score):
            continue
        # Best score in this class up to this date
        class_scores = same_class[col].dropna()
        if len(class_scores) == 0:
            continue
        best = class_scores.max()
        if score >= best:
            return True
    return False

valid['is_top'] = valid.apply(lambda r: is_top_model(r, valid), axis=1)
top_models = valid[valid['is_top']].copy()

print(f"\nTop models (frontier in param class): {len(top_models)} / {len(valid)}")
for pc in sorted(top_models['param_class'].unique()):
    subset = top_models[top_models['param_class'] == pc]
    print(f"  {pc}: {len(subset)} models")

# ── Print the data table ──────────────────────────────────────────────
print("\n" + "="*100)
print("TOP MODELS: Price/M Tokens by Parameter Class Over Time")
print("="*100)
for pc in ['<10B', '10-100B', '100B+']:
    subset = top_models[top_models['param_class'] == pc].sort_values('date')
    if len(subset) == 0:
        continue
    print(f"\n{'─'*80}")
    print(f"  Parameter Class: {pc}")
    print(f"{'─'*80}")
    print(f"  {'Date':<12} {'Model':<40} {'Params':>8} {'Price/1M tok':>14}")
    print(f"  {'─'*12} {'─'*40} {'─'*8} {'─'*14}")
    for _, r in subset.iterrows():
        name = str(r['Model'])[:40]
        print(f"  {r['date'].strftime('%Y-%m-%d'):<12} {name:<40} {r['params_B']:>7.1f}B ${r['price']:>12.4f}")

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

colors = {'<10B': '#2196F3', '10-100B': '#FF9800', '100B+': '#E91E63'}
param_classes = ['<10B', '10-100B', '100B+']

for ax, pc in zip(axes, param_classes):
    subset = top_models[top_models['param_class'] == pc].sort_values('date')
    if len(subset) == 0:
        ax.set_title(f'{pc}\n(no data)', fontsize=13)
        continue

    ax.scatter(subset['date'], subset['price'],
               c=colors[pc], s=60, alpha=0.8, zorder=5, edgecolors='white', linewidth=0.5)

    # Label each point
    for _, r in subset.iterrows():
        name = str(r['Model']).split('(')[0].strip()
        if len(name) > 20:
            name = name[:18] + '..'
        ax.annotate(name, (r['date'], r['price']),
                    fontsize=6, alpha=0.7, rotation=30,
                    textcoords='offset points', xytext=(4, 4))

    # Trend line
    if len(subset) >= 3:
        x_num = (subset['date'] - subset['date'].min()).dt.days.values.astype(float)
        y_log = np.log(subset['price'].values)
        valid_mask = np.isfinite(y_log) & np.isfinite(x_num)
        if valid_mask.sum() >= 3:
            z = np.polyfit(x_num[valid_mask], y_log[valid_mask], 1)
            x_line = np.linspace(x_num.min(), x_num.max(), 100)
            y_line = np.exp(z[1]) * np.exp(z[0] * x_line)
            date_line = subset['date'].min() + pd.to_timedelta(x_line, unit='D')
            ax.plot(date_line, y_line, '--', color=colors[pc], alpha=0.5, linewidth=2)

            # Halving time
            if z[0] < 0:
                halving_days = -np.log(2) / z[0]
                halving_months = halving_days / 30.44
                ax.text(0.05, 0.95, f'Halving: {halving_months:.1f} mo',
                        transform=ax.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_title(f'{pc} Parameters', fontsize=13, fontweight='bold')
    ax.set_xlabel('Release Date', fontsize=11)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='x', rotation=0, labelsize=8)

axes[0].set_ylabel('Blended Price (USD / 1M tokens)', fontsize=11)

fig.suptitle('Price per Million Tokens Over Time by Parameter Class\n(Frontier models only: top on ≥1 benchmark in param class at release)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/price_by_param_class.png', dpi=200, bbox_inches='tight')
print("\n\nSaved figure: figures/price_by_param_class.png")

# ── Combined overlay plot ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 7))

for pc in param_classes:
    subset = top_models[top_models['param_class'] == pc].sort_values('date')
    if len(subset) == 0:
        continue

    ax2.scatter(subset['date'], subset['price'],
                c=colors[pc], s=70, alpha=0.8, zorder=5,
                edgecolors='white', linewidth=0.5, label=pc)

    # Trend line
    if len(subset) >= 3:
        x_num = (subset['date'] - subset['date'].min()).dt.days.values.astype(float)
        y_log = np.log(subset['price'].values)
        valid_mask = np.isfinite(y_log) & np.isfinite(x_num)
        if valid_mask.sum() >= 3:
            z = np.polyfit(x_num[valid_mask], y_log[valid_mask], 1)
            x_line = np.linspace(x_num.min(), x_num.max(), 100)
            y_line = np.exp(z[1]) * np.exp(z[0] * x_line)
            date_line = subset['date'].min() + pd.to_timedelta(x_line, unit='D')
            ax2.plot(date_line, y_line, '--', color=colors[pc], alpha=0.6, linewidth=2)

ax2.set_yscale('log')
ax2.set_xlabel('Release Date', fontsize=12)
ax2.set_ylabel('Blended Price (USD / 1M tokens)', fontsize=12)
ax2.set_title('Price per Million Tokens Over Time by Parameter Class\n(Frontier models only)', fontsize=14, fontweight='bold')
ax2.legend(title='Parameter Class', fontsize=11, title_fontsize=12)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('figures/price_by_param_class_overlay.png', dpi=200, bbox_inches='tight')
print("Saved figure: figures/price_by_param_class_overlay.png")

# ── Summary statistics ────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY: Median price/1M tokens by parameter class and quarter")
print("="*80)
top_models['quarter'] = top_models['date'].dt.to_period('Q')
summary = top_models.groupby(['param_class', 'quarter'])['price'].agg(['median', 'count', 'min', 'max'])
for pc in param_classes:
    if pc not in summary.index.get_level_values(0):
        continue
    print(f"\n{pc}:")
    s = summary.loc[pc]
    for q, row in s.iterrows():
        print(f"  {q}: median=${row['median']:.4f}, range=[${row['min']:.4f}, ${row['max']:.4f}], n={int(row['count'])}")
