#!/usr/bin/env bash
#
# generate_all.sh — Regenerate benchmark CSVs, regression tables, and all figures.
#
# Usage:
#   ./generate_all.sh              # run everything
#   ./generate_all.sh --skip-csv   # skip CSV generation, only regenerate figures & tables
#
set -euo pipefail
cd "$(dirname "$0")"

# Use non-interactive matplotlib backend so no GUI windows pop up
export MPLBACKEND=Agg

SKIP_CSV=false
for arg in "$@"; do
    case "$arg" in
        --skip-csv) SKIP_CSV=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ──────────────────────────────────────────────
# Step 1: Generate per-benchmark CSVs from merged_with_training_compute.csv
# ──────────────────────────────────────────────
if [ "$SKIP_CSV" = false ]; then
    echo "=== Step 1: Generating benchmark CSVs ==="
    python3 generate_new_csv.py
    echo ""
fi

# ──────────────────────────────────────────────
# Step 2: Generate regression comparison tables (CSV + LaTeX)
# ──────────────────────────────────────────────
echo "=== Step 2: Generating regression comparison tables ==="
python3 generate_regression_tables.py
echo ""

# ──────────────────────────────────────────────
# Step 3: Generate all budget_plots_epoch figures
# ──────────────────────────────────────────────
echo "=== Step 3: Generating budget_plots_epoch figures ==="
mkdir -p figures

# Scripts use relative paths (../data/, ../figures/), so run from within the directory
pushd budget_plots_epoch > /dev/null

# Budget epoch plots
echo "  [1/12] gpqa_budget_epoch"
python3 gpqa_budget_epoch.py

echo "  [2/12] aime_budget_epoch"
python3 aime_budget_epoch.py

echo "  [3/12] swe_budget_epoch"
python3 swe_budget_epoch.py

# Pareto frontier plots
echo "  [4/12] gpqa_pareto_frontier"
python3 gpqa_pareto_frontier.py

echo "  [5/12] aime_pareto_frontier"
python3 aime_pareto_frontier.py

echo "  [6/12] swe_pareto_frontier"
python3 swe_pareto_frontier.py

# Pareto with model names
echo "  [7/12] gpqa_pareto_names"
python3 gpqa_pareto_names.py

echo "  [8/12] aime_pareto_names"
python3 aime_pareto_names.py

echo "  [9/12] swe_pareto_names"
python3 swe_pareto_names.py

# MoE vs Dense comparisons
echo "  [10/12] moe_dense_pareto (GPQA)"
python3 moe_dense_pareto_frontier.py

echo "  [11/12] moe_dense_pareto (SWE)"
python3 moe_dense_pareto_swe.py

echo "  [12/12] moe_dense_pareto (AIME)"
python3 moe_dense_pareto_aime.py

# Multi-benchmark price comparison
echo "  [13/13] multi_benchmark_price"
python3 multi_benchmark_price.py

popd > /dev/null
echo ""

# ──────────────────────────────────────────────
# Step 4: Generate total_benchmark_price figures
# ──────────────────────────────────────────────
echo "=== Step 4: Generating total_benchmark_price figures ==="
python3 total_benchmark_price.py
echo ""

# ──────────────────────────────────────────────
# Step 5: Generate bucket_figure_trends (notebook)
# ──────────────────────────────────────────────
echo "=== Step 5: Generating bucket_figure_trends ==="
jupyter nbconvert --to notebook --execute bucket_figure_trends.ipynb \
    --output /tmp/bucket_figure_trends_executed.ipynb \
    --ExecutePreprocessor.timeout=120 \
    2>&1 | tail -1
echo "  -> figures/gpqa_open_license_bucket.png, aime_open_license_bucket.png, etc."
echo ""

# ──────────────────────────────────────────────
# Step 6: Generate table_as_graph (reads regression CSV)
# ──────────────────────────────────────────────
echo "=== Step 6: Generating table_as_graph ==="
jupyter nbconvert --to notebook --execute table_as_graph.ipynb \
    --output /tmp/table_as_graph_executed.ipynb \
    --ExecutePreprocessor.timeout=120 \
    2>&1 | tail -1
echo "  -> figures/growth_rates.png"
echo ""

echo "=== Done ==="
