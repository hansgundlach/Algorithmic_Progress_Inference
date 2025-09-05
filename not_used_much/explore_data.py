import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv("art_analysis_inf_data.csv")

# Look for models with non-null token data
token_cols = [
    "input_tokens_epoch_gpqa",
    "outpur_tokens_epoch_gpqa",
    "input tokens swe",
    "output tokens swe",
]

print("Models with token data:")
for i, row in df.iterrows():
    has_token_data = False
    for col in token_cols:
        if pd.notna(row[col]) and row[col] != 0:
            has_token_data = True
            break

    if has_token_data:
        print(f"{row['Model']}: ", end="")
        for col in token_cols:
            if pd.notna(row[col]) and row[col] != 0:
                print(f"{col}={row[col]}", end=" ")
        print()

print(f"\nTotal models in dataset: {len(df)}")

# Check which price columns have data
price_date_cols = []
for col in df.columns:
    if "/" in col and ("input price" in col or "output price" in col):
        price_date_cols.append(col)

print(f"\nFound {len(price_date_cols)} price date columns")
print("Price columns:", price_date_cols[:10])  # Show first 10

# Look at a few models with actual data
print(f"\nSample data for models with token info:")
for i, row in df.iterrows():
    if pd.notna(row["input_tokens_epoch_gpqa"]) and pd.notna(
        row["outpur_tokens_epoch_gpqa"]
    ):
        print(f"\nModel: {row['Model']}")
        print(f"Input tokens: {row['input_tokens_epoch_gpqa']}")
        print(f"Output tokens: {row['outpur_tokens_epoch_gpqa']}")

        # Show some price data
        for col in price_date_cols[:6]:  # Show first 6 price columns
            if pd.notna(row[col]):
                print(f"{col}: {row[col]}")

        if i > 3:  # Only show first few examples
            break
