# Trends in Inference Efficiency

Analysis of inference price trends to measure algorithmic progress in language model inference across GPQA-D, AIME, SWE-Bench, and ARC-AGI benchmarks.

## Reproducing All Results

```bash
./generate_all.sh
```

This runs the full pipeline:

1. **Generate benchmark CSVs** from `data/merged_with_training_compute.csv` (`generate_new_csv.py`)
2. **Run regressions** and produce comparison tables (`generate_regression_tables.py`)
3. **Generate all figures** in `budget_plots_epoch/` (13 scripts: budget epochs, Pareto frontiers, model names, MoE vs dense, multi-benchmark price)
4. **Generate table-as-graph** figure (`table_as_graph.ipynb`)

Use `--skip-csv` to skip step 1 if source data hasn't changed.

## Repository Structure

### Data Pipeline
- `data/merged_with_training_compute.csv` — source data (model scores, prices over time, training compute)
- `generate_new_csv.py` — produces per-benchmark CSVs with benchmark cost history:
  - `data/gpqa_price_reduction_models.csv`
  - `data/aime_price_reduction_models.csv`
  - `data/swe_price_reduction_models.csv`
- `generate_regression_tables.py` — runs regressions across all benchmark/restriction combos, outputs to `results_data/`:
  - `regression_comparison_table_raw.csv` — raw annual cost reduction factors
  - `regression_comparison_table_hw_adjusted.csv` — adjusted for hardware progress (divided by 1/0.7)

### Figures (`budget_plots_epoch/`)
Each benchmark (GPQA-D, AIME, SWE-Bench) has:
- `*_budget_epoch.py` — best model performance over time at fixed price budgets
- `*_pareto_frontier.py` — Pareto frontier with regression trend lines
- `*_pareto_names.py` — Pareto frontier with labeled model names (ICML style)
- `moe_dense_pareto_*.py` — MoE vs dense architecture comparison
- `multi_benchmark_price.py` — cross-benchmark price comparison of best models

### Analysis Notebooks
- `table_as_graph.ipynb` — regression results as scatter plot with error bars (reads from `results_data/regression_comparison_table_raw.csv`)
- `decomposing_inf_progress.ipynb` — factor decomposition into competition, hardware, and algorithmic components
- `main_regresssion.ipynb` — interactive regression exploration

### Regression Model
`log(Price) = alpha * time + beta * logit(benchmark_score) + c`

Annual cost reduction factor = `exp(-alpha * 365)`. Reported with 90% confidence intervals.
