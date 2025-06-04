# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the CSV file
df = pd.read_csv("art_analysis_scrape.csv")


# Clean the data
# Extract numeric parameter counts
def clean_param_count(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    try:
        # Extract number and convert to billions
        if "B" in str(value):
            return float(str(value).replace("B", ""))
        else:
            return float(value) / 1e9  # Convert to billions if raw number
    except:
        return np.nan


# Clean parameter count columns
df["Parameters_B"] = df["Parameter Count"].apply(clean_param_count)
if "Parameter Count #Solid" in df.columns:
    df["Parameters_B_Solid"] = df["Parameter Count #Solid"].apply(clean_param_count)
    # Use solid parameter count if available, otherwise use regular parameter count
    df["Parameters_B_Final"] = df["Parameters_B_Solid"].fillna(df["Parameters_B"])
else:
    df["Parameters_B_Final"] = df["Parameters_B"]


# Clean price data
def clean_price(value):
    if pd.isna(value):
        return np.nan
    try:
        if isinstance(value, str) and "$" in value:
            return float(value.replace("$", "").strip())
        return float(value)
    except:
        return np.nan


df["Output_Price"] = df["Output Price\nUSD/1M Tokens"].apply(clean_price)

# Calculate price per parameter (dollars per billion parameters)
df["Price_Per_Param"] = df["Output_Price"] / df["Parameters_B_Final"]

# Filter out rows with missing data
filtered_df = df.dropna(
    subset=["Parameters_B_Final", "Output_Price", "Price_Per_Param"]
)

# Sort by parameter count for trend analysis
filtered_df = filtered_df.sort_values("Parameters_B_Final")

# Print summary statistics
print(f"Number of models with complete data: {len(filtered_df)}")
print("\nPrice per billion parameters summary statistics:")
print(filtered_df["Price_Per_Param"].describe())


# %%

# Create scatter plot with trend line
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Plot data points
sns.scatterplot(
    data=filtered_df,
    x="Parameters_B_Final",
    y="Price_Per_Param",
    hue="License",
    size="Output_Price",
    sizes=(20, 200),
    alpha=0.7,
)

# Add trend line
x = filtered_df["Parameters_B_Final"]
y = filtered_df["Price_Per_Param"]
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2)

# Add logarithmic trend line
log_x = np.log(x)
log_z = np.polyfit(log_x, y, 1)
log_p = np.poly1d(log_z)
plt.plot(x, log_p(log_x), "g-.", alpha=0.8, linewidth=2)

# Label points with model names
for i, row in filtered_df.iterrows():
    plt.text(
        row["Parameters_B_Final"] * 1.05,
        row["Price_Per_Param"] * 1.05,
        row["Model"],
        fontsize=8,
        alpha=0.7,
    )

plt.title("Price per Parameter vs. Model Size", fontsize=15)
plt.xlabel("Model Size (Billions of Parameters)", fontsize=12)
plt.ylabel("Price per Billion Parameters (USD/1M tokens)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig("price_per_parameter_trend.png", dpi=300)


# %%
# ===================================================================================


# Filter for models with MMLU-Pro between 50% and 70%
def clean_mmlu_pro(value):
    if pd.isna(value):
        return np.nan
    try:
        # Convert percentage string to float
        if isinstance(value, str) and "%" in value:
            return float(value.replace("%", ""))
        return float(value) * 100  # If already decimal, convert to percentage
    except:
        return np.nan


# Add MMLU-Pro column with cleaned values
df["MMLU_Pro"] = df["MMLU-Pro (Reasoning & Knowledge)"].apply(clean_mmlu_pro)

# Time trend analysis with MMLU-Pro filter
if "Release Date" in df.columns:
    # Convert release date to datetime
    df["Release_Date"] = pd.to_datetime(
        df["Release Date"], format="%m/%d/%y", errors="coerce"
    )

    # Filter for models with release date, parameters, price, and MMLU-Pro in range 50-70%
    time_df = df.dropna(subset=["Release_Date", "Parameters_B_Final", "Output_Price"])
    time_df = time_df[(time_df["MMLU_Pro"] >= 50) & (time_df["MMLU_Pro"] <= 70)]

    # Sort by release date
    time_df = time_df.sort_values("Release_Date")

    # Calculate price per parameter over time
    time_df["Price_Per_Param"] = time_df["Output_Price"] / time_df["Parameters_B_Final"]

    # Plot price per parameter over time
    plt.figure(figsize=(12, 8))

    # Plot data points
    sns.scatterplot(
        data=time_df,
        x="Release_Date",
        y="Price_Per_Param",
        hue="License",
        size="Parameters_B_Final",
        sizes=(20, 200),
        alpha=0.7,
    )

    # Add trend line
    try:
        # Convert dates to numeric for trend line
        x_numeric = (
            time_df["Release_Date"].astype(int) / 10**18
        )  # Normalize to avoid overflow
        z = np.polyfit(x_numeric, time_df["Price_Per_Param"], 1)
        p = np.poly1d(z)
        plt.plot(time_df["Release_Date"], p(x_numeric), "r--", alpha=0.8, linewidth=2)
    except:
        print("Could not add trend line to time plot")

    plt.title("Price per Parameter Over Time (MMLU-Pro 50-70%)", fontsize=15)
    plt.xlabel("Release Date", fontsize=12)
    plt.ylabel("Price per Billion Parameters (USD/1M tokens)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.yscale("log")

    # Save the figure with a new name to distinguish from original
    plt.savefig("price_per_parameter_time_trend_mmlu_filtered.png", dpi=300)


# %%
# Additional analysis: Parameter count vs. price
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=filtered_df, x="Parameters_B_Final", y="Output_Price", hue="License", alpha=0.7
)

# Add power law trend line (y = ax^b)
log_x = np.log(filtered_df["Parameters_B_Final"])
log_y = np.log(filtered_df["Output_Price"])
z = np.polyfit(log_x, log_y, 1)
a = np.exp(z[1])
b = z[0]
x_range = np.linspace(
    min(filtered_df["Parameters_B_Final"]), max(filtered_df["Parameters_B_Final"]), 100
)
y_pred = a * x_range**b
plt.plot(x_range, y_pred, "r--", alpha=0.8, linewidth=2)
plt.text(
    0.7,
    0.9,
    f"Power law: y = {a:.2f}x^{b:.2f}",
    transform=plt.gca().transAxes,
    fontsize=10,
)

plt.title("Output Price vs. Model Size", fontsize=15)
plt.xlabel("Model Size (Billions of Parameters)", fontsize=12)
plt.ylabel("Output Price (USD/1M tokens)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xscale("log")
plt.yscale("log")

# Save the figure
plt.savefig("parameter_count_vs_price.png", dpi=300)

print("\nAnalysis complete. Charts saved to:")
print("- price_per_parameter_trend.png")
print("- price_per_parameter_time_trend_mmlu_filtered.png")
print("- parameter_count_vs_price.png")

# %%
