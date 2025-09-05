#!/usr/bin/env python3
"""
Model Price Reduction History Generator - Blended Price Version

This script processes historical pricing data for AI models and creates a new CSV where
each price reduction for a benchmark creates a new row entry.

Key features:
1. Strict cost decrease checking (no duplicates for same model)
2. Proper price column replacement
3. Blended price column (3:1 input:output) calculation and inclusion
4. Blended price column removal for all other historical columns
5. Better duplicate detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re


def parse_date_from_column(col_name):
    """Extract date from column name like '6/1/2024 input price'"""
    match = re.search(r"(\d+/\d+/\d+)", col_name)
    if match:
        return datetime.strptime(match.group(1), "%m/%d/%Y")
    return None


def calculate_benchmark_cost(input_tokens, output_tokens, input_price, output_price):
    """Calculate total cost to run benchmark given token counts and prices per million tokens"""
    if (
        pd.isna(input_tokens)
        or pd.isna(output_tokens)
        or pd.isna(input_price)
        or pd.isna(output_price)
    ):
        return np.nan

    # Convert to cost per token (prices are per million tokens)
    input_cost = (input_tokens * input_price) / 1_000_000
    output_cost = (output_tokens * output_price) / 1_000_000

    return input_cost + output_cost


def calculate_blended_price(input_price, output_price):
    """
    Calculate the blended price using a 3:1 input:output token ratio.
    Blended price = (3 * input_price + output_price) / 4
    """
    if pd.isna(input_price) or pd.isna(output_price):
        return np.nan
    try:
        # Try to convert to float if not already
        input_price = float(input_price)
        output_price = float(output_price)
    except Exception:
        return np.nan
    return (3 * input_price + output_price) / 4


def process_price_history(df, input_token_col, output_token_col, verbose=False):
    """
    Process the dataframe to create new rows for each price reduction.
    Adds a blended price column (3:1 input:output).
    """

    # Find all price columns (input and output pairs)
    price_columns = [
        col
        for col in df.columns
        if "price" in col.lower() and ("input" in col or "output" in col)
    ]

    # Group price columns by date
    date_price_pairs = {}
    for col in price_columns:
        date = parse_date_from_column(col)
        if date:
            date_str = date.strftime("%m/%d/%Y")
            if date_str not in date_price_pairs:
                date_price_pairs[date_str] = {}

            if "input" in col:
                date_price_pairs[date_str]["input"] = col
            elif "output" in col:
                date_price_pairs[date_str]["output"] = col

    # Filter to only complete pairs (both input and output)
    complete_pairs = {
        date: cols
        for date, cols in date_price_pairs.items()
        if "input" in cols and "output" in cols
    }

    # Sort dates chronologically
    sorted_dates = sorted(
        complete_pairs.keys(), key=lambda x: datetime.strptime(x, "%m/%d/%Y")
    )

    new_rows = []
    model_price_history = {}  # Track price history per model
    global_price_combinations = set()  # Track all price combinations globally

    for idx, row in df.iterrows():
        model_name = row["Model"]
        if pd.isna(model_name) or model_name.strip() == "":
            continue

        input_tokens = row.get(input_token_col, np.nan)
        output_tokens = row.get(output_token_col, np.nan)

        # Skip if we don't have token data
        if pd.isna(input_tokens) or pd.isna(output_tokens):
            continue

        # Initialize price history for this model
        if model_name not in model_price_history:
            model_price_history[model_name] = []

        for date_str in sorted_dates:
            input_price_col = complete_pairs[date_str]["input"]
            output_price_col = complete_pairs[date_str]["output"]

            input_price = row.get(input_price_col, np.nan)
            output_price = row.get(output_price_col, np.nan)

            # Skip if prices are not available
            if pd.isna(input_price) or pd.isna(output_price):
                continue

            # Skip if prices are zero (likely missing data)
            if input_price == 0 or output_price == 0:
                continue

            current_cost = calculate_benchmark_cost(
                input_tokens, output_tokens, input_price, output_price
            )

            # Skip if cost calculation failed
            if pd.isna(current_cost):
                continue

            # Check if this is the lowest cost we've seen for this model
            previous_costs = [
                entry["cost"] for entry in model_price_history[model_name]
            ]
            min_previous_cost = min(previous_costs) if previous_costs else float("inf")

            epsilon = 1e-10
            if current_cost < (min_previous_cost - epsilon):
                # Check if this exact price combination already exists ANYWHERE (globally)
                price_combo = (input_price, output_price)

                if price_combo in global_price_combinations:
                    if verbose:
                        print(
                            f"  {model_name} - {date_str}: Skipping duplicate price combination ${input_price}/${output_price} (already exists globally)"
                        )
                    continue

                if verbose:
                    print(
                        f"  {model_name} - {date_str}: Cost decreased from ${min_previous_cost:.6f} to ${current_cost:.6f}"
                    )

                # Create new row
                new_row = row.copy()

                # Update model name to include date
                new_row["Model"] = f"{model_name} {date_str}"

                # Update release date
                new_row["Release Date"] = date_str

                # Update the ONLY two price columns that should exist
                new_row["Input Price\nUSD/1M Tokens"] = input_price
                new_row["Output Price\nUSD/1M Tokens"] = output_price

                # Add the calculated benchmark cost
                new_row["Benchmark Cost USD"] = current_cost

                # Add the blended price (3:1 input:output)
                blended_price = calculate_blended_price(input_price, output_price)
                # Ensure the blended price is a float, not an object or string
                if pd.isna(blended_price):
                    new_row["Blended Price (3:1) USD/1M Tokens"] = np.nan
                else:
                    new_row["Blended Price (3:1) USD/1M Tokens"] = float(blended_price)

                # Remove ALL other price-related columns
                price_cols_to_remove = [
                    "Lowest Output Price Found AA",
                    "Lowest Input Price AA",
                    "Lowest Blended Price AA",
                    "price input lowest gpr",
                    "price output lowest gpqa",
                    "total price lowest gpqa",
                    "total price swe",
                ]

                # Remove blended price columns (historical)
                blended_cols = [
                    col for col in new_row.index if "blended" in col.lower()
                ]
                price_cols_to_remove.extend(blended_cols)

                # Remove all historical price columns (the time series ones)
                price_cols_to_remove.extend(price_columns)

                for col in price_cols_to_remove:
                    if col in new_row:
                        new_row[col] = np.nan

                new_rows.append(new_row)

                # Track this entry in price history
                model_price_history[model_name].append(
                    {"cost": current_cost, "prices": price_combo, "date": date_str}
                )

                # Add to global price combinations to prevent future duplicates
                global_price_combinations.add(price_combo)

            elif verbose and current_cost != float("inf"):
                if abs(current_cost - min_previous_cost) < epsilon:
                    print(
                        f"  {model_name} - {date_str}: Cost unchanged (${current_cost:.6f})"
                    )
                else:
                    print(
                        f"  {model_name} - {date_str}: Cost increased from ${min_previous_cost:.6f} to ${current_cost:.6f}"
                    )

    if new_rows:
        result_df = pd.DataFrame(new_rows)
        result_df.reset_index(drop=True, inplace=True)

        # Ensure blended price column is recalculated and filled if missing
        if "Blended Price (3:1) USD/1M Tokens" not in result_df.columns:
            result_df["Blended Price (3:1) USD/1M Tokens"] = np.nan

        # Recalculate blended price for all rows to ensure it's not empty
        result_df["Blended Price (3:1) USD/1M Tokens"] = result_df.apply(
            lambda r: calculate_blended_price(
                r.get("Input Price\nUSD/1M Tokens", np.nan),
                r.get("Output Price\nUSD/1M Tokens", np.nan),
            ),
            axis=1,
        )

        # Keep ONLY these price-related columns, remove all others
        keep_price_cols = [
            "Input Price\nUSD/1M Tokens",
            "Output Price\nUSD/1M Tokens",
            "Benchmark Cost USD",
            "Blended Price (3:1) USD/1M Tokens",
        ]

        # Find all price/cost-related columns to potentially drop
        all_price_cols = [
            col
            for col in result_df.columns
            if "price" in col.lower() or "cost" in col.lower()
        ]

        # Drop price columns that are NOT in our keep list
        cols_to_drop = [col for col in all_price_cols if col not in keep_price_cols]

        if cols_to_drop:
            result_df = result_df.drop(columns=cols_to_drop)

        # Ensure the blended price column is float and not object
        result_df["Blended Price (3:1) USD/1M Tokens"] = pd.to_numeric(
            result_df["Blended Price (3:1) USD/1M Tokens"], errors="coerce"
        )

        return result_df
    else:
        # If no new rows, return an empty DataFrame with the expected columns
        return pd.DataFrame(
            columns=[
                "Model",
                "Release Date",
                "Input Price\nUSD/1M Tokens",
                "Output Price\nUSD/1M Tokens",
                "Benchmark Cost USD",
                "Blended Price (3:1) USD/1M Tokens",
            ]
        )


def main():
    """Main function to process the CSV file"""

    # Configuration
    INPUT_FILE = "inference_data_new_large.csv"
    OUTPUT_FILE = "price_reduction_models.csv"

    # Benchmark token columns
    INPUT_TOKEN_COL = "input_tokens_epoch_gpqa"
    OUTPUT_TOKEN_COL = "output_tokens_epoch_gpqa"

    # Set to True to see detailed processing information
    VERBOSE = False

    print("=" * 60)
    print("Model Price Reduction History Generator (BLENDED PRICE)")
    print("=" * 60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Benchmark: {INPUT_TOKEN_COL} + {OUTPUT_TOKEN_COL}")
    print("=" * 60)

    try:
        # Read the CSV file
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} rows")

        # Check if required columns exist
        if INPUT_TOKEN_COL not in df.columns:
            print(f"Error: Input token column '{INPUT_TOKEN_COL}' not found")
            return

        if OUTPUT_TOKEN_COL not in df.columns:
            print(f"Error: Output token column '{OUTPUT_TOKEN_COL}' not found")
            return

        print(f"Processing price history...")

        # Process the data
        result_df = process_price_history(
            df, INPUT_TOKEN_COL, OUTPUT_TOKEN_COL, verbose=VERBOSE
        )

        print(f"Generated {len(result_df)} rows with price reductions")

        if len(result_df) > 0:
            # Save the result
            result_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved results to {OUTPUT_FILE}")

            # Check for duplicates in result
            print("\nChecking for duplicate price combinations:")
            result_prices = result_df[
                ["Model", "Input Price\nUSD/1M Tokens", "Output Price\nUSD/1M Tokens"]
            ].copy()

            # Group by model base name
            model_groups = {}
            for idx, row in result_prices.iterrows():
                model_base = row["Model"].rsplit(" ", 1)[0]  # Remove date
                price_combo = (
                    row["Input Price\nUSD/1M Tokens"],
                    row["Output Price\nUSD/1M Tokens"],
                )

                if model_base not in model_groups:
                    model_groups[model_base] = []
                model_groups[model_base].append((row["Model"], price_combo))

            duplicates_found = False
            for model_base, entries in model_groups.items():
                if len(entries) > 1:
                    prices_seen = set()
                    for model_full, price_combo in entries:
                        if price_combo in prices_seen:
                            print(
                                f"  DUPLICATE: {model_full} has duplicate prices: {price_combo}"
                            )
                            duplicates_found = True
                        prices_seen.add(price_combo)

            if not duplicates_found:
                print("  ✓ No duplicate price combinations found!")

            print(
                f"\nProcessing complete! Generated {len(result_df)} unique model entries."
            )
        else:
            print("No price reductions found in the data")

    except FileNotFoundError:
        print(f"Error: Could not find file {INPUT_FILE}")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
