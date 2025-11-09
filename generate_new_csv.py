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
6. Optional cache token cost calculation
7. Optional reasoning token cost calculation (conditionally added to output tokens)
8. Reasoning_in_output flag to control whether reasoning tokens are already in output or need to be added
9. Cache_in_input and cache_in_output flags to control whether cache tokens are included in token counts
10. Works with or without token count columns (with warning when not specified)
11. Error handling for mismatched token/price availability
12. Validation that reasoning_in_output is specified when reasoning tokens are present

Cost Formula:
    ((input_tokens * input_price) +
     ((output_tokens [+ reasoning_tokens if not in output]) * output_price) +
     (cache_read * cache_read_price) +
     (cache_write * cache_write_price)) * epoch_normalization

Notes:
- Epoch normalization: All costs are normalized to 1 epoch using: normalized_cost = cost * (1 / epochs)
- If cache_in_input=True: cache_read tokens are already included in input_tokens, so we subtract them
  and charge separately: (input_tokens - cache_read) * input_price + cache_read * cache_read_price
- If cache_in_input=False: cache tokens are separate, use: input_tokens * input_price + cache_read * cache_read_price
- Same logic applies for cache_in_output with cache_write tokens
- Reasoning tokens are only added to output if reasoning_in_output=False.
  If reasoning_in_output=True, reasoning tokens are already included in output_tokens.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import os


def parse_date_from_column(col_name):
    """Extract date from column name like '6/1/2024 input price'"""
    match = re.search(r"(\d+/\d+/\d+)", col_name)
    if match:
        return datetime.strptime(match.group(1), "%m/%d/%Y")
    return None


def calculate_benchmark_cost(
    input_tokens,
    output_tokens,
    input_price,
    output_price,
    reasoning_tokens=None,
    reasoning_in_output=None,
    cache_read_tokens=None,
    cache_write_tokens=None,
    cache_read_price=None,
    cache_write_price=None,
    cache_in_input=None,
    cache_in_output=None,
):
    """
    Calculate total cost to run benchmark given token counts and prices per million tokens.
    Formula: (input_tokens * input_price + output_tokens * output_price +
              cache_read * cache_read_price + cache_write * cache_write_price)

    With adjustments based on flags:
    - If cache_in_input=True: (input_tokens - cache_read) * input_price + cache_read * cache_read_price
    - If cache_in_output=True: (output_tokens - cache_write) * output_price + cache_write * cache_write_price

    Args:
        reasoning_tokens: Optional reasoning tokens (defaults to 0)
        reasoning_in_output: Boolean flag indicating if reasoning tokens are already included in output_tokens.
                           If True, reasoning tokens are NOT added to output (already included).
                           If False, reasoning tokens ARE added to output.
                           Must not be None/NA if reasoning_tokens is provided.
        cache_in_input: Boolean flag indicating if cache_read tokens are already included in input_tokens.
                       If True, subtract cache_read from input_tokens before applying input_price.
                       If False/None, cache tokens are separate.
        cache_in_output: Boolean flag indicating if cache_write tokens are already included in output_tokens.
                        If True, subtract cache_write from output_tokens before applying output_price.
                        If False/None, cache tokens are separate.

    Raises:
        ValueError: If input_price is available but output_price is not, or vice versa
        ValueError: If input_tokens is available but output_tokens is not, or vice versa
        ValueError: If reasoning_tokens is provided but reasoning_in_output is None/NA
    """
    # Check for mismatches in token numbers
    has_input_tokens = not pd.isna(input_tokens)
    has_output_tokens = not pd.isna(output_tokens)

    if has_input_tokens != has_output_tokens:
        raise ValueError(
            f"Token number mismatch: input_tokens={'available' if has_input_tokens else 'missing'}, "
            f"output_tokens={'available' if has_output_tokens else 'missing'}. "
            f"Both must be available or both must be missing."
        )

    # Check for mismatches in token prices
    has_input_price = not pd.isna(input_price)
    has_output_price = not pd.isna(output_price)

    if has_input_price != has_output_price:
        raise ValueError(
            f"Token price mismatch: input_price={'available' if has_input_price else 'missing'}, "
            f"output_price={'available' if has_output_price else 'missing'}. "
            f"Both must be available or both must be missing."
        )

    # If any required value is missing, return NaN
    if (
        not has_input_tokens
        or not has_output_tokens
        or not has_input_price
        or not has_output_price
    ):
        return np.nan

    # Convert string values to numeric, handling any formatting issues
    try:
        input_tokens = float(input_tokens)
        output_tokens = float(output_tokens)
        input_price = float(input_price)
        output_price = float(output_price)
    except (ValueError, TypeError):
        return np.nan

    # Handle reasoning tokens (default to 0 if not provided)
    reasoning_tokens_value = 0.0
    add_reasoning_to_output = False

    if reasoning_tokens is not None and not pd.isna(reasoning_tokens):
        try:
            reasoning_tokens_value = float(reasoning_tokens)

            # If reasoning tokens are present, reasoning_in_output must be specified
            if reasoning_tokens_value > 0:
                if reasoning_in_output is None or pd.isna(reasoning_in_output):
                    raise ValueError(
                        f"reasoning_in_output must be specified (True/False) when reasoning_tokens is provided (reasoning_tokens={reasoning_tokens_value})"
                    )

                # If reasoning_in_output is False, we need to ADD reasoning tokens to output
                # If reasoning_in_output is True, reasoning tokens are already in output, so don't add
                if reasoning_in_output == False:
                    add_reasoning_to_output = True

        except (ValueError, TypeError) as e:
            if "reasoning_in_output" in str(e):
                raise  # Re-raise validation errors
            reasoning_tokens_value = 0.0

    # Handle cache tokens (default to 0 if not provided)
    cache_read_value = 0.0
    cache_write_value = 0.0
    cache_read_price_value = 0.0
    cache_write_price_value = 0.0

    # Check if we have cache tokens and prices
    has_cache = (
        cache_read_tokens is not None
        and cache_write_tokens is not None
        and cache_read_price is not None
        and cache_write_price is not None
        and not pd.isna(cache_read_tokens)
        and not pd.isna(cache_write_tokens)
        and not pd.isna(cache_read_price)
        and not pd.isna(cache_write_price)
    )

    if has_cache:
        try:
            cache_read_value = float(cache_read_tokens)
            cache_write_value = float(cache_write_tokens)
            cache_read_price_value = float(cache_read_price)
            cache_write_price_value = float(cache_write_price)
        except (ValueError, TypeError):
            # If cache conversion fails, treat as no cache (values stay at 0)
            pass

    # Calculate cost using the formula:
    # Base: input_tokens * input_price + output_tokens * output_price +
    #       cache_read * cache_read_price + cache_write * cache_write_price
    #
    # With adjustments:
    # - If cache_in_input=True: (input_tokens - cache_read) * input_price + cache_read * cache_read_price
    # - If cache_in_output=True: (output_tokens - cache_write) * output_price + cache_write * cache_write_price

    # Determine input tokens to charge at input_price
    input_tokens_for_price = input_tokens
    if cache_in_input == True and cache_read_value > 0:
        # Cache tokens are included in input_tokens, so subtract them
        input_tokens_for_price = input_tokens - cache_read_value

    input_cost = (input_tokens_for_price * input_price) / 1_000_000

    # Determine output tokens to charge at output_price
    # Only add reasoning tokens to output if they're not already included
    reasoning_adjustment = reasoning_tokens_value if add_reasoning_to_output else 0.0
    output_tokens_for_price = output_tokens + reasoning_adjustment

    if cache_in_output == True and cache_write_value > 0:
        # Cache tokens are included in output_tokens, so subtract them
        output_tokens_for_price = output_tokens_for_price - cache_write_value

    output_cost = (output_tokens_for_price * output_price) / 1_000_000

    # Always charge cache tokens at cache prices
    cache_read_cost = (cache_read_value * cache_read_price_value) / 1_000_000
    cache_write_cost = (cache_write_value * cache_write_price_value) / 1_000_000

    total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

    return total_cost


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


def process_price_history(
    df,
    input_token_col=None,
    output_token_col=None,
    reasoning_token_col=None,
    reasoning_in_output_col=None,
    cache_read_token_col=None,
    cache_write_token_col=None,
    cache_read_price_col=None,
    cache_write_price_col=None,
    cache_in_input_col=None,
    cache_in_output_col=None,
    epoch_column=None,
    verbose=False,
):
    """
    Process the dataframe to create new rows for each price reduction.
    Adds a blended price column (3:1 input:output).
    Optionally includes reasoning tokens (conditionally added to output tokens in cost calculation).
    Optionally includes cache token costs if cache columns are provided.
    Optionally normalizes benchmark costs to 16 epochs using epoch_column.

    Can work with or without token columns - if token columns are not provided,
    it will track price reductions based on blended price only.

    Args:
        reasoning_token_col: Optional column name containing reasoning tokens.
                            Defaults to 0 if not provided.
        reasoning_in_output_col: Optional column name indicating if reasoning tokens are already included
                                in output tokens (True) or need to be added (False).
                                Required if reasoning_token_col is specified and has non-zero values.
        cache_in_input_col: Optional column name indicating if cache_read tokens are already included
                           in input_tokens (True) or are separate (False).
        cache_in_output_col: Optional column name indicating if cache_write tokens are already included
                            in output_tokens (True) or are separate (False).
        epoch_column: Optional column name containing the number of epochs.
                     All costs will be normalized to 1 epoch using the formula:
                     normalized_cost = cost * (1 / epochs)
    """

    # Check if token columns are provided
    use_token_cost = input_token_col is not None and output_token_col is not None

    if not use_token_cost:
        print("WARNING: Input and output token number columns not specified.")
        print(
            "Will track price reductions based on blended price only (no benchmark cost calculation)."
        )
    else:
        # Check if token columns exist in dataframe
        missing_token_cols = []
        for col in [input_token_col, output_token_col]:
            if col not in df.columns:
                missing_token_cols.append(col)

        if missing_token_cols:
            print(f"Warning: Token columns not found in data: {missing_token_cols}")
            print(
                "Will track price reductions based on blended price only (no benchmark cost calculation)."
            )
            use_token_cost = False

    # Check if epoch column is provided and exists
    use_epoch_normalization = False
    if epoch_column:
        if epoch_column not in df.columns:
            print(f"Warning: Epoch column '{epoch_column}' not found in data.")
            print("Proceeding without epoch normalization...")
        else:
            use_epoch_normalization = True
            print(f"Using epoch normalization with column: {epoch_column}")
            print("All benchmark costs will be normalized to 1 epoch.")

    # Check cache configuration
    use_cache = all(
        [
            cache_read_token_col,
            cache_write_token_col,
            cache_read_price_col,
            cache_write_price_col,
        ]
    )

    if use_cache and use_token_cost:
        # Verify cache columns exist in dataframe
        missing_cache_cols = []
        for col in [
            cache_read_token_col,
            cache_write_token_col,
            cache_read_price_col,
            cache_write_price_col,
        ]:
            if col not in df.columns:
                missing_cache_cols.append(col)

        if missing_cache_cols:
            print(f"Warning: Cache columns not found in data: {missing_cache_cols}")
            print("Proceeding without cache token calculation...")
            use_cache = False
        else:
            print(f"Using cache token calculation with columns:")
            print(f"  Cache read tokens: {cache_read_token_col}")
            print(f"  Cache write tokens: {cache_write_token_col}")
            print(f"  Cache read price: {cache_read_price_col}")
            print(f"  Cache write price: {cache_write_price_col}")
    else:
        use_cache = False
        if (
            cache_read_token_col
            or cache_write_token_col
            or cache_read_price_col
            or cache_write_price_col
        ):
            print(
                "Cache columns specified but not using token cost calculation or missing token columns."
            )
            print("Cache calculation disabled.")

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

    for idx, row in df.iterrows():
        model_name = row["Model"]
        if pd.isna(model_name) or model_name.strip() == "":
            continue

        # Get token data if using token cost calculation
        if use_token_cost:
            input_tokens = row.get(input_token_col, np.nan)
            output_tokens = row.get(output_token_col, np.nan)

            # Skip if we don't have token data when using token cost
            if pd.isna(input_tokens) or pd.isna(output_tokens):
                continue

            # Get reasoning tokens (defaults to 0 if not provided)
            reasoning_tokens = (
                row.get(reasoning_token_col, np.nan) if reasoning_token_col else None
            )

            # Get reasoning_in_output flag
            reasoning_in_output = (
                row.get(reasoning_in_output_col, np.nan)
                if reasoning_in_output_col
                else None
            )

            # Get cache token data if using cache
            cache_read_tokens = (
                row.get(cache_read_token_col, np.nan) if use_cache else None
            )
            cache_write_tokens = (
                row.get(cache_write_token_col, np.nan) if use_cache else None
            )
            cache_read_price = (
                row.get(cache_read_price_col, np.nan) if use_cache else None
            )
            cache_write_price = (
                row.get(cache_write_price_col, np.nan) if use_cache else None
            )

            # Get cache_in_input and cache_in_output flags
            cache_in_input = (
                row.get(cache_in_input_col, np.nan) if cache_in_input_col else None
            )
            cache_in_output = (
                row.get(cache_in_output_col, np.nan) if cache_in_output_col else None
            )
        else:
            input_tokens = output_tokens = np.nan
            reasoning_tokens = None
            reasoning_in_output = None
            cache_read_tokens = cache_write_tokens = None
            cache_read_price = cache_write_price = None
            cache_in_input = cache_in_output = None

        # Get epoch data if using epoch normalization
        epochs = None
        epoch_normalization_factor = 1.0
        if use_epoch_normalization:
            epochs = row.get(epoch_column, np.nan)
            if pd.isna(epochs) or epochs <= 0:
                if verbose:
                    print(
                        f"  {model_name}: Missing or invalid epoch data, skipping row"
                    )
                continue  # Skip this row entirely if epoch data is missing when epoch column is provided

            try:
                epochs = float(epochs)
                epoch_normalization_factor = 1.0 / epochs
                if verbose:
                    print(
                        f"  {model_name}: Using {epochs} epochs, normalization factor: {epoch_normalization_factor:.4f}"
                    )
            except (ValueError, TypeError):
                if verbose:
                    print(
                        f"  {model_name}: Invalid epoch value '{epochs}', skipping row"
                    )
                continue  # Skip this row if epoch value cannot be converted

        # --- BEGIN: Warn if tokens specified but price missing (per model/row) ---
        # For each date, check if tokens are specified but price is missing for input/output/cache
        for date_str in sorted_dates:
            input_price_col = complete_pairs[date_str]["input"]
            output_price_col = complete_pairs[date_str]["output"]

            input_price = row.get(input_price_col, np.nan)
            output_price = row.get(output_price_col, np.nan)

            # Input tokens & price
            if (
                input_token_col
                and input_token_col in row
                and not pd.isna(row[input_token_col])
            ):
                if pd.isna(input_price):
                    print(
                        f"WARNING: Model '{model_name}' specifies input tokens ({input_token_col}={row[input_token_col]}) but is missing input price for {date_str} ({input_price_col})"
                    )

            # Output tokens & price
            if (
                output_token_col
                and output_token_col in row
                and not pd.isna(row[output_token_col])
            ):
                if pd.isna(output_price):
                    print(
                        f"WARNING: Model '{model_name}' specifies output tokens ({output_token_col}={row[output_token_col]}) but is missing output price for {date_str} ({output_price_col})"
                    )

            # Cache read tokens & price
            if (
                cache_read_token_col
                and cache_read_token_col in row
                and not pd.isna(row[cache_read_token_col])
            ):
                if cache_read_price_col and (
                    cache_read_price_col not in row
                    or pd.isna(row.get(cache_read_price_col, np.nan))
                ):
                    print(
                        f"WARNING: Model '{model_name}' specifies cache read tokens ({cache_read_token_col}={row[cache_read_token_col]}) but is missing cache read price ({cache_read_price_col}) for {date_str}"
                    )

            # Cache write tokens & price
            if (
                cache_write_token_col
                and cache_write_token_col in row
                and not pd.isna(row[cache_write_token_col])
            ):
                if cache_write_price_col and (
                    cache_write_price_col not in row
                    or pd.isna(row.get(cache_write_price_col, np.nan))
                ):
                    print(
                        f"WARNING: Model '{model_name}' specifies cache write tokens ({cache_write_token_col}={row[cache_write_token_col]}) but is missing cache write price ({cache_write_price_col}) for {date_str}"
                    )
        # --- END: Warn if tokens specified but price missing ---

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

            # Calculate comparison metric (benchmark cost if available, otherwise blended price)
            if use_token_cost:
                try:
                    current_metric = calculate_benchmark_cost(
                        input_tokens,
                        output_tokens,
                        input_price,
                        output_price,
                        reasoning_tokens,
                        reasoning_in_output,
                        cache_read_tokens,
                        cache_write_tokens,
                        cache_read_price,
                        cache_write_price,
                        cache_in_input,
                        cache_in_output,
                    )
                except ValueError as e:
                    # Raise error if there's a mismatch in token/price availability
                    raise ValueError(
                        f"Error processing model '{model_name}' at date {date_str}: {str(e)}"
                    )
                # Apply epoch normalization if enabled
                if use_epoch_normalization and not pd.isna(current_metric):
                    current_metric = current_metric * epoch_normalization_factor
                metric_name = "cost"
            else:
                current_metric = calculate_blended_price(input_price, output_price)
                metric_name = "blended price"

            # Skip if metric calculation failed
            if pd.isna(current_metric):
                continue

            # Check if this is the lowest metric we've seen for this model
            previous_metrics = [
                entry["metric"] for entry in model_price_history[model_name]
            ]
            min_previous_metric = (
                min(previous_metrics) if previous_metrics else float("inf")
            )

            epsilon = 1e-10
            if current_metric < (min_previous_metric - epsilon):
                # Check if this exact price combination already exists for THIS MODEL ONLY
                price_combo = (input_price, output_price)

                # Check if this model already has this price combination
                model_price_combos = [
                    entry["prices"] for entry in model_price_history[model_name]
                ]

                if price_combo in model_price_combos:
                    if verbose:
                        print(
                            f"  {model_name} - {date_str}: Skipping duplicate price combination ${input_price}/${output_price} (already exists for this model)"
                        )
                    continue

                if verbose:
                    print(
                        f"  {model_name} - {date_str}: {metric_name.capitalize()} decreased from ${min_previous_metric:.6f} to ${current_metric:.6f}"
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

                # Add the calculated benchmark cost (if available)
                if use_token_cost:
                    new_row["Benchmark Cost USD"] = current_metric
                else:
                    new_row["Benchmark Cost USD"] = np.nan

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

                # Don't remove cache-related columns (we want to keep them for viewing)
                cache_cols = [
                    col
                    for col in price_cols_to_remove
                    if "cache" in col.lower()
                    and ("read" in col.lower() or "write" in col.lower())
                ]
                for cache_col in cache_cols:
                    if cache_col in price_cols_to_remove:
                        price_cols_to_remove.remove(cache_col)

                for col in price_cols_to_remove:
                    if col in new_row:
                        new_row[col] = np.nan

                new_rows.append(new_row)

                # Track this entry in price history
                model_price_history[model_name].append(
                    {"metric": current_metric, "prices": price_combo, "date": date_str}
                )

            elif verbose and current_metric != float("inf"):
                if abs(current_metric - min_previous_metric) < epsilon:
                    print(
                        f"  {model_name} - {date_str}: {metric_name.capitalize()} unchanged (${current_metric:.6f})"
                    )
                else:
                    print(
                        f"  {model_name} - {date_str}: {metric_name.capitalize()} increased from ${min_previous_metric:.6f} to ${current_metric:.6f}"
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

        # Keep cache-related columns (for viewing)
        cache_cols_to_keep = [
            col
            for col in all_price_cols
            if "cache" in col.lower()
            and ("read" in col.lower() or "write" in col.lower())
        ]

        # Add cache columns to the keep list
        keep_price_cols.extend(cache_cols_to_keep)

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
        # Note: Cache columns will be added if they exist in the data
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


def process_configuration(config, df, verbose=False):
    """
    Process a single configuration and generate the output CSV.

    Args:
        config: Dictionary containing configuration parameters
        df: DataFrame with the input data
        verbose: Whether to show detailed processing information

    Returns:
        True if successful, False otherwise
    """
    config_name = config["name"]
    input_file = config["input_file"]
    output_file = config["output_file"]
    input_token_col = config["input_token_col"]
    output_token_col = config["output_token_col"]
    reasoning_token_col = config.get("reasoning_token_col")
    reasoning_in_output_col = config.get("reasoning_in_output_col")
    cache_read_token_col = config.get("cache_read_token_col")
    cache_write_token_col = config.get("cache_write_token_col")
    cache_read_price_col = config.get("cache_read_price_col")
    cache_write_price_col = config.get("cache_write_price_col")
    cache_in_input_col = config.get("cache_in_input_col")
    cache_in_output_col = config.get("cache_in_output_col")
    epoch_column = config.get("epoch_column")

    print("\n" + "=" * 60)
    print(f"Processing {config_name}")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    if input_token_col and output_token_col:
        print(f"Benchmark: {input_token_col} + {output_token_col}")
    else:
        print("Benchmark: Blended price tracking only (no token columns specified)")
    print("=" * 60)

    try:
        # Check if required columns exist (only if they're specified)
        input_token_col_actual = input_token_col
        output_token_col_actual = output_token_col

        if input_token_col and input_token_col not in df.columns:
            print(f"Warning: Input token column '{input_token_col}' not found in data")
            input_token_col_actual = None

        if output_token_col and output_token_col not in df.columns:
            print(
                f"Warning: Output token column '{output_token_col}' not found in data"
            )
            output_token_col_actual = None

        print(f"Processing price history...")

        # Process the data
        result_df = process_price_history(
            df,
            input_token_col_actual,
            output_token_col_actual,
            reasoning_token_col,
            reasoning_in_output_col,
            cache_read_token_col,
            cache_write_token_col,
            cache_read_price_col,
            cache_write_price_col,
            cache_in_input_col,
            cache_in_output_col,
            epoch_column,
            verbose=verbose,
        )

        print(f"Generated {len(result_df)} rows with price reductions")

        if len(result_df) > 0:
            # Count unique models (disregarding dates)
            unique_models = set()
            for model_name in result_df["Model"]:
                # Remove date suffix to get base model name
                base_model = model_name.rsplit(" ", 1)[0]
                unique_models.add(base_model)

            print(
                f"Number of unique models (disregarding price change dates): {len(unique_models)}"
            )

            # Ensure data directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created directory: {output_dir}")

            # Save the result
            result_df.to_csv(output_file, index=False)
            print(f"Saved results to {output_file}")

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
                f"\n{config_name} processing complete! Generated {len(result_df)} unique model entries."
            )
            return True
        else:
            print(f"No price reductions found in the data for {config_name}")
            return False

    except Exception as e:
        print(f"Error processing {config_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function to process all configurations"""

    # Define all configurations
    configurations = [
        {
            "name": "GPQA-Diamond",
            "input_file": "data/inference_data_new_large.csv",
            "output_file": "data/price_reduction_models.csv",
            "input_token_col": "input_tokens_epoch_gpqa",
            "output_token_col": "output_tokens_epoch_gpqa",
            "reasoning_token_col": None,
            "reasoning_in_output_col": "gqpa_reasoning_in_output",
            "cache_read_token_col": None,
            "cache_write_token_col": None,
            "cache_read_price_col": None,
            "cache_write_price_col": None,
            "cache_in_input_col": None,
            "cache_in_output_col": None,
            "epoch_column": "gpqa_epochs",
        },
        {
            "name": "SWE-Bench",
            "input_file": "data/inference_data_new_large.csv",
            "output_file": "data/swe_price_reduction_models.csv",
            "input_token_col": "input tokens swe",
            "output_token_col": "output tokens swe",
            "reasoning_token_col": None,
            "reasoning_in_output_col": None,
            "cache_read_token_col": "cache reads swe",
            "cache_write_token_col": "cache write swe",
            "cache_read_price_col": "cache_read_cost",
            "cache_write_price_col": "cache_write_cost",
            "cache_in_input_col": "cache_in_input",
            "cache_in_output_col": "cache_in_output",
            "epoch_column": None,
        },
        {
            "name": "AIME",
            "input_file": "data/inference_data_new_large.csv",
            "output_file": "data/aime_price_reduction_models.csv",
            "input_token_col": "input tokens AIME",
            "output_token_col": "output tokens AIME",
            "reasoning_token_col": "AIME_reasoning",
            "reasoning_in_output_col": "reasoning_in_output",
            "cache_read_token_col": "cache read tokens aiml",
            "cache_write_token_col": "cache write tokens aiml",
            "cache_read_price_col": "cache read cost aiml",
            "cache_write_price_col": "cache write cost aiml",
            "cache_in_input_col": None,
            "cache_in_output_col": None,
            "epoch_column": "AIME_epochs",
        },
    ]

    # Set to True to see detailed processing information
    VERBOSE = False

    print("=" * 60)
    print("Model Price Reduction History Generator (BLENDED PRICE)")
    print("Processing all configurations: GPQA, SWE-Bench, and AIME")
    print("=" * 60)

    # Read the input CSV file once (all configurations use the same input file)
    input_file = configurations[0]["input_file"]
    try:
        df = pd.read_csv(input_file)
        print(f"\nLoaded {len(df)} rows from {input_file}")
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
        return
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        import traceback

        traceback.print_exc()
        return

    # Process each configuration
    results = []
    for config in configurations:
        success = process_configuration(config, df, verbose=VERBOSE)
        results.append((config["name"], success))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
