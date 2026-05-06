"""
dataset_stats.py
----------------
Quickly reports, for each benchmark dataset:
  1. Total rows in the CSV
  2. Unique models  (base model name, ignoring the date suffix that
     `generate_new_csv.py` appends — each row is a price-reduction
     event, so one model can appear multiple times)
  3. (Model, date, price) points — rows with valid Model, Release
     Date, and Benchmark Cost USD > 0.
  4. Open-source breakdown (License == "Open"): unique base models
     and number of (model, date, price) data points.

Usage:
    python dataset_stats.py
    python dataset_stats.py --gpqa /path/to/gpqa.csv ...
"""

import argparse
import re
import pandas as pd


BENCHMARKS = [
    {
        "name": "GPQA-Diamond",
        "default_path": "data/gpqa_price_reduction_models.csv",
        "arg": "gpqa",
    },
    {
        "name": "SWE-bench Verified",
        "default_path": "data/swe_price_reduction_models.csv",
        "arg": "swe",
    },
    {
        "name": "AIME",
        "default_path": "data/aime_price_reduction_models.csv",
        "arg": "aime",
    },
]

# `generate_new_csv.py` writes Model as f"{base_name} {MM/DD/YYYY}" for every
# price-reduction event. Strip that trailing date to recover the base model.
DATE_SUFFIX_RE = re.compile(r"\s+\d{1,2}/\d{1,2}/\d{2,4}\s*$")


def base_model_name(name: object) -> object:
    if not isinstance(name, str):
        return name
    return DATE_SUFFIX_RE.sub("", name).strip()


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
    df["Benchmark Cost USD"] = pd.to_numeric(
        df["Benchmark Cost USD"].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )
    df["Base Model"] = df["Model"].map(base_model_name)
    return df


def report(df: pd.DataFrame, bench: dict) -> None:
    name = bench["name"]
    total_rows = len(df)
    unique_models = df["Base Model"].dropna().nunique()

    complete_mask = (
        df["Model"].notna()
        & df["Release Date"].notna()
        & df["Benchmark Cost USD"].notna()
        & (df["Benchmark Cost USD"] > 0)
    )
    n_points = int(complete_mask.sum())
    unique_models_complete = df.loc[complete_mask, "Base Model"].dropna().nunique()

    # ── Open-source breakdown (License column == "Open") ────────────────────
    if "License" in df.columns:
        license_norm = df["License"].astype(str).str.strip().str.lower()
        open_mask = license_norm == "open"

        open_unique_models = (
            df.loc[open_mask, "Base Model"].dropna().nunique()
        )
        open_points = int((open_mask & complete_mask).sum())
        open_unique_models_complete = (
            df.loc[open_mask & complete_mask, "Base Model"].dropna().nunique()
        )

        # Unknown / missing license entries (helpful sanity check)
        n_unknown_license = int(
            (~license_norm.isin(["open", "proprietary"])).sum()
        )
    else:
        open_unique_models = open_points = open_unique_models_complete = 0
        n_unknown_license = total_rows

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)
    print(f"  Total rows in CSV                  : {total_rows}")
    print(f"  Unique base models (all rows)      : {unique_models}")
    print(f"  Unique base models (valid pts)     : {unique_models_complete}")
    print(f"  (Model, date, price) data points   : {n_points}")
    print(f"  ── Open-source ──")
    print(f"  Open-source unique models (all)    : {open_unique_models}")
    print(f"  Open-source unique models (valid)  : {open_unique_models_complete}")
    print(f"  Open-source data points            : {open_points}")
    if n_unknown_license:
        print(f"  Rows with unknown/missing license  : {n_unknown_license}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics for benchmark CSVs")
    for b in BENCHMARKS:
        parser.add_argument(
            f"--{b['arg']}",
            default=b["default_path"],
            help=f"Path to {b['name']} CSV (default: {b['default_path']})",
        )
    args = parser.parse_args()

    paths = {b["arg"]: getattr(args, b["arg"]) for b in BENCHMARKS}

    print("\n" + "=" * 60)
    print("  BENCHMARK DATASET STATISTICS")
    print("=" * 60)

    for bench in BENCHMARKS:
        path = paths[bench["arg"]]
        try:
            df = load(path)
        except FileNotFoundError:
            print(f"\n[!] File not found: {path}  (skipping {bench['name']})")
            continue
        report(df, bench)

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
