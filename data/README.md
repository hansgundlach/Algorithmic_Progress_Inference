---
license: cc-by-4.0
language:
  - en
tags:
  - ai-evaluation
  - benchmarks
  - llm
  - inference-cost
  - price-performance
  - gpqa
  - swe-bench
  - aime
  - algorithmic-progress
pretty_name: "The Price of Progress: Benchmark-Level LLM Inference Cost Dataset"
size_categories:
  - n<1K
task_categories:
  - text-generation
---

# The Price of Progress: Benchmark-Level LLM Inference Cost Dataset

## Dataset Summary

This dataset combines historical LLM inference prices with benchmark performance scores to construct the largest publicly available benchmark-level LLM price dataset we are aware of. It covers **100+ models** across three major benchmarks (**GPQA-Diamond**, **SWE-bench Verified**, and **AIME**) over a two-year window from **April 2024 to April 2026**, with varying coverage per benchmark.

The dataset was created to support analysis of how AI capability diffuses over time on a *per-dollar* basis — distinguishing price-independent technical progress from progress driven purely by larger, more expensive models. It is released alongside an anonymized NeurIPS 2026 submission.

> **"The Price of Progress: Revisiting Benchmark Progress in AI"**  
> Anonymous Authors — NeurIPS 2026 submission (under review)

### Key statistics

| Benchmark | Price data points | Unique models |
|---|---|---|
| GPQA-Diamond | 166 | 115 |
| AIME (OTIS Mock AIME 2024–2025) | 138 | ~100 |
| SWE-bench Verified | 31 | 29 |

---

## Dataset Structure

### File

`combined_benchmark_price_data.csv` — all three benchmarks in a single file, distinguished by the `Benchmark` column.

### Column Descriptions

| Column | Type | Description |
|---|---|---|
| `Benchmark` | string | Benchmark name: `GPQA-Diamond`, `AIME`, or `SWE-Bench Verified` |
| `Model` | string | Model identifier, typically including a date suffix (MM/YYYY) reflecting the price-snapshot date |
| `Release Date` | date (MM/DD/YYYY) | The date of the price snapshot (not necessarily model release) |
| `Benchmark Score` | float (%) | Model accuracy on the benchmark as a percentage |
| `Benchmark Price USD` | float | Estimated total cost in USD to run the full benchmark on this model at this snapshot date |
| `Input Price USD/1M Tokens` | float | Input token price per million tokens at snapshot date (USD) |
| `Output Price USD/1M Tokens` | float | Output token price per million tokens at snapshot date (USD) |
| `Cache Read Price USD/1M Tokens` | float | Cache read token price per million tokens (USD); may be empty |
| `Cache Write Price USD/1M Tokens` | float | Cache write token price per million tokens (USD); may be empty |
| `Input Tokens` | float | Number of input tokens used to run the benchmark |
| `Output Tokens` | float | Number of output tokens generated |
| `Reasoning Tokens` | float | Number of reasoning tokens (thinking tokens), if applicable |
| `Reasoning In Output` | boolean | Whether reasoning tokens are counted as part of output tokens |
| `Cache Read Tokens` | float | Number of cache-read tokens used |
| `Cache Write Tokens` | float | Number of cache-write tokens used |
| `Cache In Input` | boolean | Whether cache read tokens are counted as part of input tokens |
| `Cache In Output` | boolean | Whether cache write tokens are counted as part of output tokens |

### Benchmark Price Computation

`Benchmark Price USD` is computed by multiplying the relevant token counts by the corresponding historical token prices:

```
Benchmark Price = (Input Tokens × Input Price) + (Output Tokens × Output Price)
                + (Cache Read Tokens × Cache Read Price) + (Cache Write Tokens × Cache Write Price)
```

Token counts are normalized to a single benchmark run (divided by the number of runs if Epoch AI ran the benchmark multiple times).

---

## Data Sources

**Token prices** are collected from [Artificial Analysis](https://artificialanalysis.ai) via the [Internet Archive (Wayback Machine)](https://web.archive.org). We record the lowest available input and output price across all inference providers at each snapshot date. Cache token prices for proprietary models are sourced directly from the model providers (e.g., Anthropic, OpenAI, DeepSeek).

**Benchmark scores and token usage** are sourced from [Epoch AI's LLM Benchmarking Hub](https://epoch.ai/data/llm-benchmarking-hub), which reports model-level performance along with input, output, reasoning, and cached token counts per benchmark run.

---

## License and Upstream Terms

This dataset combines author-created derived fields with upstream benchmark and pricing data.

**Author-created metadata**, cleaning decisions, derived variables, and documentation are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Analysis code is released separately under the [MIT License](https://opensource.org/licenses/MIT).

**Epoch AI Benchmarking Hub data** is used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with attribution to [Epoch AI](https://epoch.ai).

**Artificial Analysis price data** is attributed to [Artificial Analysis](https://artificialanalysis.ai). We do not claim ownership of Artificial Analysis source fields or relicense them. Artificial Analysis-derived fields remain subject to Artificial Analysis's applicable terms and policies.

Users should attribute both this dataset and the relevant upstream sources when using the data.

| Asset | License / Terms |
|---|---|
| Author-created metadata, derived variables, cleaning decisions, documentation | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Analysis code (separate repository) | [MIT License](https://opensource.org/licenses/MIT) |
| Epoch AI Benchmarking Hub data (upstream) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — used with attribution to Epoch AI |
| Artificial Analysis price data (upstream) | Attributed to Artificial Analysis; not relicensed — subject to Artificial Analysis's applicable terms and policies |

---

## Dataset Creation

### Motivation

Benchmark leaderboards report raw accuracy scores, but do not account for the cost of inference. A model that achieves higher accuracy by running 100× more tokens is not necessarily a more practical advance than a cheaper model with slightly lower accuracy. This dataset makes it possible to evaluate AI progress on a *per-dollar* basis and to study the economic forces shaping LLM inference costs over time.

### Data Collection Procedure

1. **Price data**: Internet Archive snapshots of Artificial Analysis pages were scraped to reconstruct a historical time series of input/output token prices for each model-provider pair. We retain only the lowest available price at each snapshot to reflect the accessible cost frontier.
2. **Benchmark data**: Token usage and benchmark scores were downloaded from the Epoch AI LLM Benchmarking Hub.
3. **Matching**: Model names were manually matched between Artificial Analysis and Epoch AI entries. Entries that could not be matched were excluded.
4. **Price computation**: Benchmark prices were computed by multiplying token counts by historical prices. Models with zero-dollar cost entries (promotional offers) were excluded.
5. **Temporal treatment**: Price changes over time for the same model are treated as separate data points.

### Preprocessing Notes

- Models with input or output cost of \$0 are excluded (typically promotional offers).
- Cache tokens are excluded from GPQA-Diamond cost estimates (they are ~20× smaller than input/output tokens and Artificial Analysis does not provide historical cache prices), but are included for SWE-bench Verified where they constitute a substantial share of cost.
- For SWE-bench Verified, cache prices are taken from current provider pricing (vendor cache prices rarely change for a given model version).
- Benchmark token counts are normalized to a single run by dividing by the number of Epoch AI evaluation runs.
- Multiple reasoning-budget variants of the same model (e.g., Claude 3.7 Sonnet at different reasoning levels) are treated as distinct models.

---

## Uses

### Intended Uses

- Measuring benchmark price-performance improvement rates over time (conditional on accuracy).
- Estimating algorithmic efficiency progress in LLM inference.
- Constructing cost-performance Pareto frontiers.
- Studying the decomposition of price changes into hardware, algorithmic, and competitive effects.
- Analyzing how much benchmark progress is associated with rising inference expenditure vs. price-independent technical gains.

### Out-of-Scope Uses

- Precise marginal cost estimation (this dataset reflects user-facing prices, not provider marginal costs).
- Benchmarks not covered (GPQA-Diamond, AIME, SWE-bench Verified only).
- Periods before April 2024, where historical price data is sparse.

---

## Limitations

- **Benchmark coverage**: Only three benchmarks are included. Price trends may differ for other domains.
- **Price data gaps**: Internet Archive coverage is uneven; some models have price snapshots only months apart, which can affect trend estimates.
- **User-facing prices**: Prices reflect publicly available inference API prices, not provider costs. Latency, rate limits, and enterprise contracts are not captured.
- **Cache price history**: We do not have historical cache token price data; current prices are used as proxies.
- **Small open-weight subsets**: The open-weight model subset is small, especially for SWE-bench Verified (n=9), making open-weight-specific conclusions less certain.
- **Decomposition is suggestive**: The hardware/algorithmic/competitive decomposition is approximate and should not be interpreted causally.

---

## Citation

This dataset accompanies an anonymized NeurIPS 2026 submission. A full citation will be provided upon de-anonymization after review. In the meantime, please cite the upstream data sources if you use this dataset:

```bibtex
@misc{EpochLLMBenchmarkingHub2024,
  title        = {LLM Benchmarking Hub},
  author       = {{Epoch AI}},
  year         = {2024},
  howpublished = {\url{https://epoch.ai/data/llm-benchmarking-hub}}
}

@misc{artificialanalysis,
  title        = {Artificial Analysis},
  author       = {{Artificial Analysis}},
  howpublished = {\url{https://artificialanalysis.ai}}
}
```

---

## Acknowledgements

We thank Epoch AI for making benchmark data available under CC BY 4.0 and Artificial Analysis for making inference price data publicly accessible.
