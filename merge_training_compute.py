"""
Merge training compute data from gpqa_diamond.csv into merged.csv

This script:
1. Loads training compute data from gpqa_diamond.csv
2. Performs fuzzy matching between model names
3. Adds three new columns to merged.csv:
   - Training_Compute_FLOP: The training compute value
   - training_compute_matched_model: The model name from gpqa_diamond.csv that was matched
   - training_compute_match_score: The fuzzy match score (0-1)
   - training_compute_matched: Boolean indicating if a match was found
"""

import pandas as pd
import numpy as np
from thefuzz import fuzz
from thefuzz import process

# Load data
print("Loading data...")
merged_df = pd.read_csv('data/merged.csv')
gpqa_df = pd.read_csv('data/gpqa_diamond.csv')

print(f"Merged CSV: {len(merged_df)} rows")
print(f"GPQA Diamond: {len(gpqa_df)} rows")

# Extract model names and training compute
gpqa_models = gpqa_df[['Model version', 'Training compute (FLOP)']].copy()
gpqa_models.columns = ['model_name', 'training_compute']
gpqa_models = gpqa_models.dropna(subset=['training_compute'])
gpqa_models = gpqa_models[gpqa_models['training_compute'] > 0]

print(f"GPQA models with training compute: {len(gpqa_models)}")

# Create a dictionary for fast lookup
compute_dict = dict(zip(gpqa_models['model_name'], gpqa_models['training_compute']))

print("\nPerforming fuzzy matching...")

# Initialize new columns
merged_df['Training_Compute_FLOP'] = np.nan
merged_df['training_compute_matched_model'] = ''
merged_df['training_compute_match_score'] = np.nan
merged_df['training_compute_matched'] = False

# Match threshold
MATCH_THRESHOLD = 80  # 80% similarity required

matched_count = 0
for idx, row in merged_df.iterrows():
    model_name = str(row['Model'])

    if pd.isna(model_name) or model_name == '' or model_name == 'nan':
        continue

    # Try exact match first
    if model_name in compute_dict:
        merged_df.at[idx, 'Training_Compute_FLOP'] = compute_dict[model_name]
        merged_df.at[idx, 'training_compute_matched_model'] = model_name
        merged_df.at[idx, 'training_compute_match_score'] = 100.0
        merged_df.at[idx, 'training_compute_matched'] = True
        matched_count += 1
        continue

    # Try fuzzy matching
    match = process.extractOne(model_name, gpqa_models['model_name'].tolist(),
                              scorer=fuzz.token_sort_ratio)

    if match and match[1] >= MATCH_THRESHOLD:
        matched_model = match[0]
        match_score = match[1]

        merged_df.at[idx, 'Training_Compute_FLOP'] = compute_dict[matched_model]
        merged_df.at[idx, 'training_compute_matched_model'] = matched_model
        merged_df.at[idx, 'training_compute_match_score'] = match_score
        merged_df.at[idx, 'training_compute_matched'] = True
        matched_count += 1

        if match_score < 95:  # Log imperfect matches
            print(f"  Matched '{model_name}' → '{matched_model}' (score: {match_score})")

print(f"\n✓ Successfully matched {matched_count} models")
print(f"✗ Unmatched: {len(merged_df) - matched_count} models")

# Save to new file
output_file = 'data/merged_with_training_compute.csv'
merged_df.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# Show summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
matched_df = merged_df[merged_df['training_compute_matched'] == True]
print(f"\nModels with training compute: {len(matched_df)}")
print(f"Training compute range: {matched_df['Training_Compute_FLOP'].min():.2e} to {matched_df['Training_Compute_FLOP'].max():.2e} FLOP")
print(f"Average match score: {matched_df['training_compute_match_score'].mean():.1f}")

# Show some examples
print("\n" + "="*80)
print("EXAMPLE MATCHES")
print("="*80)
example_matches = matched_df[['Model', 'Training_Compute_FLOP', 'training_compute_matched_model', 'training_compute_match_score']].head(10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(example_matches.to_string(index=False))

print("\n✓ Done!")
