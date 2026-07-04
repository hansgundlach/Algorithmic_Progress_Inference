#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path


TABLE1_INPUT = "results_data/regression_comparison_table_raw.csv"
HARDWARE_ADJUSTED_INPUT = "results_data/regression_comparison_table_hw_adjusted.csv"
TABLE2_INPUT = "results_data/multi_benchmark_logit_regression.csv"
TABLE1_OUTPUT = "results_data/regression_results_paper.tex"
HARDWARE_ADJUSTED_OUTPUT = "results_data/regression_results_hardware_adjusted_paper.tex"
TABLE2_OUTPUT = "results_data/multi_benchmark_regression_paper.tex"
APPENDIX_TABLES_OUTPUT = "results_data/appendix_data_tables_paper.tex"
TABLE1_FACTOR_MULTIPLIER = 1.0

TABLE1_CAPTION = (
    "Rate of price change across several different benchmarks using general "
    "regression approach. Regressions include either all models or only the "
    "models that improve in accuracy or price (Pareto Restricted). A separate "
    "analysis of only open weight models was possible with GPQA-Diamond "
    "(GPQA-Diamond), OTIS-MOCK AIME 2024-2025 (AIME), and SWE-Bench "
    "Verified (SWE-V). Decrease factors "
    "$<1$ represent increases."
)
TABLE2_CAPTION = (
    "Here we show the logit performance trend for all models as well as Pareto "
    "and Frontier models (as defined in the paper) with and without price controls."
)
HARDWARE_ADJUSTED_CAPTION = (
    r"Annual reduction factor (hardware-adjusted) and 90\% CI (hardware-adjusted)."
)
HARDWARE_ADJUSTED_TEXT = (
    r"Trends in benchmark price-performance are also influenced by hardware "
    r"performance trends. If we want to isolate the component due purely to "
    r"algorithmic advances, we have to divide the annual factor decrease by "
    r"the annual hardware price-efficiency gain. Here we use our general "
    r"regression approach and estimates from \citet{epoch2024priceperformancehardware}, "
    r"which finds that for a fixed performance level, costs have dropped by "
    r"$30\%$ a year."
)

TABLE1_BENCHMARK_ORDER = ["GPQA", "AIME", "SWE-Bench"]
TABLE1_BENCHMARK_DISPLAY = {
    "GPQA": "GPQA-Diamond",
    "AIME": "AIME",
    "SWE-Bench": r"SWE\mbox{-}V",
}

TABLE2_BENCHMARK_ORDER = ["GPQA", "SWE-Bench", "AIME"]
TABLE2_BENCHMARK_DISPLAY = {
    "GPQA": "GPQA-Diamond",
    "GPQA-D": "GPQA-Diamond",
    "SWE-Bench": "SWE-Bench",
    "AIME": "AIME",
}

RESTRICTION_ORDER = [
    "Pareto Restricted All License",
    "Pareto Restricted Open License",
    "All License (no restriction)",
    "Open License (no restriction)",
]
RESTRICTION_DISPLAY = {
    "Pareto Restricted All License": "Pareto Restricted All License",
    "Pareto Restricted Open License": "Pareto Restricted Open Weight",
    "All License (no restriction)": "All License (no restriction)",
    "Open License (no restriction)": "Open Weight (no restriction)",
}

SAMPLE_ORDER = [
    ("Pareto", "Without Price Control"),
    ("Pareto", "With Price Control"),
    ("Frontier", "Without Price Control"),
    ("Frontier", "With Price Control"),
    ("All", "Without Price Control"),
    ("All", "With Price Control"),
]


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fill_benchmark_blanks(rows):
    current = ""
    for row in rows:
        benchmark = row.get("Benchmark", "").strip()
        if benchmark:
            current = benchmark
        else:
            row["Benchmark"] = current
    return rows


def fmt_int(value):
    try:
        n = int(float(str(value).strip()))
    except ValueError:
        return "---"
    return str(n) if n > 0 else "---"


def fmt_float(value, digits):
    try:
        x = float(str(value).strip())
    except ValueError:
        return "---"
    if not math.isfinite(x):
        return "---"
    return f"{x:.{digits}f}"


def fmt_coef(value, digits):
    rendered = fmt_float(value, digits)
    if rendered == "---":
        return rendered
    if rendered.startswith("-"):
        return f"${rendered}$"
    return rendered


def parse_ci(ci_text):
    text = str(ci_text).strip()
    match = re.fullmatch(r"\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]", text)
    if not match:
        return None
    try:
        lower = float(match.group(1))
        upper = float(match.group(2))
    except ValueError:
        return None
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    if lower < 0 or upper <= 0 or upper < lower:
        return None
    if upper > 1000:
        return None
    return lower, upper


def row_prefix(index, benchmark_label):
    if index == 0:
        return benchmark_label
    if index % 2 == 1:
        return r"\rowcolor{restrgray} \cellcolor{white}"
    return r"\cellcolor{white}"


def table1_lookup(rows):
    filled = fill_benchmark_blanks(rows)
    return {
        (row["Benchmark"].strip(), row["Restriction"].strip()): row
        for row in filled
    }


def render_table1(
    rows,
    multiplier,
    *,
    table_environment=r"\begin{table*}[h!]",
    caption=TABLE1_CAPTION,
    label="tab:regression_results",
    include_small=False,
    adjusted_headers=False,
    arraystretch="1.2",
    label_after_tabular=False,
):
    lookup = table1_lookup(rows)
    lines = [
        rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}",
        "",
        table_environment,
        r"\centering",
        rf"\caption{{{caption}}}",
        "",
    ]
    if include_small:
        lines.append(r"\small")
    lines.extend([
        r"\begin{tabular}{|l|l|c|c|c|c|}",
        r"\hline",
    ])
    if adjusted_headers:
        lines.extend([
            r"\textbf{Benchmark} & \textbf{Restriction} &",
            r"\shortstack{\textbf{Annual reduction factor}\\\textbf{(hardware-adjusted)}} &",
            r"\shortstack{\textbf{90\% CI}\\\textbf{(hardware-adjusted)}} &",
            r"\textbf{n} & \textbf{$R^2$} \\",
        ])
    else:
        lines.append(
            r"\textbf{Benchmark} & \textbf{Restriction} & \textbf{Annual Reduction Factor} & \textbf{90\% CI} & \textbf{n} & \textbf{$R^2$} \\"
        )
    lines.extend([r"\hline", ""])

    for benchmark in TABLE1_BENCHMARK_ORDER:
        benchmark_label = TABLE1_BENCHMARK_DISPLAY[benchmark]
        comment_label = benchmark_label.replace(r"\mbox{-}", "-")
        lines.append(rf"% ===== {comment_label} =====")
        for i, restriction in enumerate(RESTRICTION_ORDER):
            row = lookup.get((benchmark, restriction))
            factor = ci = n = r2 = "---"
            ci = "[---, ---]"

            if row:
                parsed_ci = parse_ci(row.get("90% CI", ""))
                raw_factor = fmt_float(row.get("Year Decrease Factor", ""), 12)
                if parsed_ci and raw_factor != "---":
                    factor_value = float(raw_factor) * multiplier
                    lower, upper = parsed_ci
                    factor = f"{factor_value:.3f}"
                    ci = f"[{lower * multiplier:.3f}, {upper * multiplier:.3f}]"
                    n = fmt_int(row.get("n", ""))
                    r2 = fmt_float(row.get("R²", row.get("R^2", "")), 4)

            line = (
                f"{row_prefix(i, benchmark_label)} & "
                f"{RESTRICTION_DISPLAY[restriction]} & {factor} & {ci} & {n} & {r2} \\\\"
            )
            if i < len(RESTRICTION_ORDER) - 1:
                line += r" \cline{2-6}"
            lines.append(line)
        lines.append(r"\hline")
        lines.append("")

    end_environment = table_environment.replace(r"\begin", r"\end").split("[", 1)[0]
    lines.append(r"\end{tabular}")
    if label_after_tabular:
        lines.extend(["", rf"\label{{{label}}}"])
    else:
        lines.insert(5, rf"\label{{{label}}}")
    lines.extend([end_environment, ""])
    return "\n".join(lines)


def render_hardware_adjusted_table(rows):
    return render_table1(
        rows,
        1.0,
        table_environment=r"\begin{table}[h!]",
        caption=HARDWARE_ADJUSTED_CAPTION,
        label="tab:regression_results_adjusted",
        include_small=True,
        adjusted_headers=True,
        arraystretch="1.15",
    )


def table2_lookup(rows):
    lookup = {}
    for row in rows:
        benchmark = row["Benchmark"].strip()
        if benchmark == "GPQA-D":
            benchmark = "GPQA"

        if "Model" in row:
            control = row["Model"].strip()
        else:
            control = (
                "With Price Control"
                if row.get("Price Control", "").strip().lower() in {"yes", "true", "1"}
                else "Without Price Control"
            )
        lookup[(benchmark, row["Sample"].strip(), control)] = row
    return lookup


def get_table2_value(row, canonical, *fallbacks):
    for key in (canonical, *fallbacks):
        if key in row:
            return row[key]
    return ""


def render_table2(rows):
    lookup = table2_lookup(rows)
    lines = [
        r"\renewcommand{\arraystretch}{1.2}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{TABLE2_CAPTION}}}",
        r"\label{tab:multi_benchmark_regression_rounded}",
        r"\renewcommand{\arraystretch}{1.2}",
        "",
        r"\begin{tabular}{|l|l|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{Benchmark} &",
        r"\textbf{Sample} &",
        r"\textbf{n} &",
        r"\shortstack{\textbf{Time}\\\textbf{Coef}} &",
        r"\shortstack{\textbf{Time}\\\textbf{SE}} &",
        r"\shortstack{\textbf{Price}\\\textbf{Coef}} &",
        r"\textbf{$R^2$} \\",
        r"\hline",
        "",
    ]

    for benchmark in TABLE2_BENCHMARK_ORDER:
        benchmark_label = TABLE2_BENCHMARK_DISPLAY[benchmark]
        lines.append(rf"% ===== {benchmark_label} =====")
        for i, (sample, model) in enumerate(SAMPLE_ORDER):
            row = lookup.get((benchmark, sample, model))
            sample_label = f"{sample}, {model}"
            n = time_coef = time_se = price_coef = r2 = "---"

            if row:
                n = fmt_int(get_table2_value(row, "n", "N"))
                time_coef = fmt_coef(
                    get_table2_value(row, "Time Coef (per year)", "Time Coef"),
                    2,
                )
                time_se = fmt_coef(
                    get_table2_value(row, "Time SE (per year)", "Time SE"),
                    3,
                )
                if model == "With Price Control":
                    price_coef = fmt_coef(
                        get_table2_value(row, "Price Coef (log price)", "Price Coef"),
                        3,
                    )
                r2 = fmt_float(row.get("R²", row.get("R^2", "")), 2)

            line = (
                f"{row_prefix(i, benchmark_label)} & {sample_label} & {n} & "
                f"{time_coef} & {time_se} & {price_coef} & {r2} \\\\"
            )
            if i < len(SAMPLE_ORDER) - 1:
                line += r" \cline{2-7}"
            lines.append(line)
        lines.append(r"\hline")
        lines.append("")

    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def render_appendix_data_tables(table1, hardware_adjusted_table, table2):
    lines = [
        r"\section{Data Tables}",
        "",
        table1.rstrip(),
        "",
        r"\FloatBarrier",
        r"\subsection{Adjusting for GPU Price-Performance Gains}",
        HARDWARE_ADJUSTED_TEXT,
        "",
        hardware_adjusted_table.rstrip(),
        "",
        r"\FloatBarrier",
        "",
        table2.rstrip(),
        "",
    ]
    return "\n".join(lines)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render paper-ready LaTeX tables from checked-in result CSVs."
    )
    parser.add_argument("--table1-input", default=TABLE1_INPUT)
    parser.add_argument("--hardware-adjusted-input", default=HARDWARE_ADJUSTED_INPUT)
    parser.add_argument("--table2-input", default=TABLE2_INPUT)
    parser.add_argument("--table1-output", default=TABLE1_OUTPUT)
    parser.add_argument("--hardware-adjusted-output", default=HARDWARE_ADJUSTED_OUTPUT)
    parser.add_argument("--table2-output", default=TABLE2_OUTPUT)
    parser.add_argument("--appendix-tables-output", default=APPENDIX_TABLES_OUTPUT)
    parser.add_argument(
        "--table1-multiplier",
        type=float,
        default=TABLE1_FACTOR_MULTIPLIER,
        help="Multiplier applied to Table 1 reduction factors and CI endpoints.",
    )
    args = parser.parse_args()

    table1_rows = read_rows(args.table1_input)
    table1 = render_table1(table1_rows, args.table1_multiplier)
    hardware_adjusted_table = render_hardware_adjusted_table(
        read_rows(args.hardware_adjusted_input)
    )
    table2 = render_table2(read_rows(args.table2_input))

    write_text(args.table1_output, table1)
    write_text(args.hardware_adjusted_output, hardware_adjusted_table)
    write_text(args.table2_output, table2)
    write_text(
        args.appendix_tables_output,
        render_appendix_data_tables(table1, hardware_adjusted_table, table2),
    )


if __name__ == "__main__":
    main()
