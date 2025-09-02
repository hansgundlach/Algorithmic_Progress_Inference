import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('art_analysis_inf_data.csv')

# Display basic info about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names (first 30):")
for i, col in enumerate(df.columns[:30]):
    print(f"{i}: {col}")

print("\nColumn names containing 'price' or date patterns:")
for i, col in enumerate(df.columns):
    if 'price' in col.lower() or '/' in col or any(month in col for month in ['1/2024', '2/2024', '3/2024', '4/2024', '5/2024', '6/2024', '7/2024', '8/2024', '9/2024', '10/2024', '11/2024', '12/2024', '1/2025', '2/2025', '3/2025']):
        print(f"{i}: {col}")

# Look at some sample data
print("\nFirst few rows of relevant columns:")
print(df[['Model']].head())

