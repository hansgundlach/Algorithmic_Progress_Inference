#!/usr/bin/env python3
"""Combine generated benchmark price-reduction CSVs into one dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BENCHMARK_INPUTS = [
    {
        "benchmark": "GPQA-Diamond",
        "benchmark_short": "GPQA",
        "score_column": "epoch_gpqa",
        "input_tokens_column": "input_tokens_epoch_gpqa",
        "output_tokens_column": "output_tokens_epoch_gpqa",
        "reasoning_tokens_column": "gpqa_reasoning_tokens",
        "reasoning_in_output_column": "gpqa_reasoning_in_output",
        "cache_read_tokens_column": "cache_read_gpqa",
        "cache_write_tokens_column": "cache_output_gpqa",
        "cache_read_price_column": None,
        "cache_write_price_column": None,
        "cache_in_input_column": None,
        "cache_in_output_column": None,
        "path": Path("data/gpqa_price_reduction_models.csv"),
    },
    {
        "benchmark": "AIME",
        "benchmark_short": "AIME",
        "score_column": "oneshot_AIME",
        "input_tokens_column": "input tokens AIME",
        "output_tokens_column": "output tokens AIME",
        "reasoning_tokens_column": "AIME_reasoning",
        "reasoning_in_output_column": "reasoning_in_output",
        "cache_read_tokens_column": "cache reads AIME",
        "cache_write_tokens_column": None,
        "cache_read_price_column": None,
        "cache_write_price_column": None,
        "cache_in_input_column": None,
        "cache_in_output_column": None,
        "path": Path("data/aime_price_reduction_models.csv"),
    },
    {
        "benchmark": "SWE-Bench Verified",
        "benchmark_short": "SWE-Bench",
        "score_column": "epoch_swe",
        "input_tokens_column": "input tokens swe",
        "output_tokens_column": "output tokens swe",
        "reasoning_tokens_column": "reasoning swe",
        "reasoning_in_output_column": "reasoning_in_output_swe",
        "cache_read_tokens_column": "cache reads swe",
        "cache_write_tokens_column": "cache write swe",
        "cache_read_price_column": "cache_read_cost",
        "cache_write_price_column": "cache_write_cost",
        "cache_in_input_column": "cache_in_input",
        "cache_in_output_column": "cache_in_output",
        "path": Path("data/swe_price_reduction_models.csv"),
    },
]

OUTPUT_PATH = Path("data/combined_benchmark_price_data.csv")


def _series_or_na(df: pd.DataFrame, column: object) -> pd.Series:
    if not column:
        return pd.NA
    column = str(column)
    if column not in df.columns:
        return pd.NA
    return df[column]


def _numeric_series_or_na(df: pd.DataFrame, column: object) -> pd.Series:
    series = _series_or_na(df, column)
    if not isinstance(series, pd.Series):
        return series
    return pd.to_numeric(
        series.astype(str).str.replace("[$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def read_benchmark_csv(config: dict[str, object]) -> pd.DataFrame:
    path = config["path"]
    if not isinstance(path, Path):
        raise TypeError("benchmark config path must be a pathlib.Path")
    if not path.exists():
        raise FileNotFoundError(f"Missing generated benchmark CSV: {path}")

    df = pd.read_csv(path)
    score_column = str(config["score_column"])
    if score_column not in df.columns:
        raise ValueError(f"{path} is missing expected score column: {score_column}")
    if "Benchmark Cost USD" not in df.columns:
        raise ValueError(f"{path} is missing expected total cost column: Benchmark Cost USD")

    return pd.DataFrame(
        {
            "Benchmark": str(config["benchmark"]),
            "Model": df["Model"],
            "Release Date": df["Release Date"],
            "Benchmark Score": df[score_column],
            "Benchmark Price USD": _numeric_series_or_na(df, "Benchmark Cost USD"),
            "Input Price USD/1M Tokens": _numeric_series_or_na(
                df, "Input Price\nUSD/1M Tokens"
            ),
            "Output Price USD/1M Tokens": _numeric_series_or_na(
                df, "Output Price\nUSD/1M Tokens"
            ),
            "Cache Read Price USD/1M Tokens": _numeric_series_or_na(
                df, config.get("cache_read_price_column")
            ),
            "Cache Write Price USD/1M Tokens": _numeric_series_or_na(
                df, config.get("cache_write_price_column")
            ),
            "Input Tokens": _numeric_series_or_na(df, config.get("input_tokens_column")),
            "Output Tokens": _numeric_series_or_na(df, config.get("output_tokens_column")),
            "Reasoning Tokens": _numeric_series_or_na(
                df, config.get("reasoning_tokens_column")
            ),
            "Reasoning In Output": _series_or_na(
                df, config.get("reasoning_in_output_column")
            ),
            "Cache Read Tokens": _numeric_series_or_na(
                df, config.get("cache_read_tokens_column")
            ),
            "Cache Write Tokens": _numeric_series_or_na(
                df, config.get("cache_write_tokens_column")
            ),
            "Cache In Input": _series_or_na(df, config.get("cache_in_input_column")),
            "Cache In Output": _series_or_na(df, config.get("cache_in_output_column")),
        }
    )


def main() -> None:
    frames = [read_benchmark_csv(config) for config in BENCHMARK_INPUTS]
    combined = pd.concat(frames, ignore_index=True, sort=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    counts = combined["Benchmark"].value_counts(sort=False)
    print(f"Wrote {OUTPUT_PATH} with {len(combined)} rows")
    for benchmark, count in counts.items():
        print(f"  {benchmark}: {count}")


if __name__ == "__main__":
    main()
