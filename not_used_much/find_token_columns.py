import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv("art_analysis_inf_data.csv")

print("Column names containing 'token':")
for i, col in enumerate(df.columns):
    if "token" in col.lower():
        print(f"{i}: {col}")

print("\nSample data for some potential input/output token columns:")
# Look at some likely token columns based on the pattern you mentioned
potential_token_cols = []
for col in df.columns:
    if any(
        x in col.lower()
        for x in ["input_tokens", "output_tokens", "tokens_epoch", "tokens_swe"]
    ):
        potential_token_cols.append(col)

for col in potential_token_cols[:10]:  # Show first 10 to avoid too much output
    print(f"\n{col}:")
    print(df[col].head())
