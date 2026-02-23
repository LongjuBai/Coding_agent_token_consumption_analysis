"""
Model Token Usage Comparison

This script compares token usage across different LLM models.
It calculates and visualizes each model's average token cost across all instances.

Two comparison modes:
1. All instances: Compare token usage across all 500 instances
2. Success subset: Compare token usage only for instances where all 8 models 
   have at least one successful run, using only successful runs for each model

Approach:
1. Load data from multiple models
2. For each model, calculate average token usage across all instances and runs
3. Find success subset: instances where all models have at least one success
4. For success subset, calculate average token cost using only successful runs
5. Visualize model comparisons using bar plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent

def discover_model_paths() -> dict:
    """Discover model CSV paths under dataset root."""
    model_paths = {}
    pattern = "*/swe_bench_token_cost_aggregated_total_with_accuracy.csv"
    for csv_path in sorted(DATASET_ROOT.glob(pattern)):
        model_dir = csv_path.parent.name
        model_paths[model_dir] = csv_path
    return model_paths


MODEL_PATHS = discover_model_paths()

# Accuracy columns for each run
acc_cols = [f"acc_run{i}" for i in range(1, 5)]

# Set larger font sizes globally
plt.rcParams.update({'font.size': 18})

print("Loading data from multiple models...")
# Load and combine data from all models
all_data = []
for model_name, csv_path in MODEL_PATHS.items():
    if Path(csv_path).exists():
        df_model = pd.read_csv(csv_path)
        df_model['model'] = model_name
        all_data.append(df_model)
        print(f"  Loaded {len(df_model)} instances from {model_name}")
    else:
        print(f"  Warning: File not found for {model_name}: {csv_path}")

if not all_data:
    raise ValueError("No data files found!")

# Combine all models
df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal instances across all models: {len(df)}")
print(f"Models included: {df['model'].unique()}\n")

# Step 1: Find success subset - instances where all 8 models have at least one successful run
print("=" * 80)
print("Step 1: Finding success subset (instances where all models have at least one success)...")
print("=" * 80)

# Use all discovered models
all_models = sorted(MODEL_PATHS.keys())
successful_instances = set()

# Check each instance
for instance_id in df['instance_id'].unique():
    instance_data = df[df['instance_id'] == instance_id]
    
    # Check if all models have at least one successful run for this instance
    all_models_have_success = True
    for model_name in all_models:
        model_instance_data = instance_data[instance_data['model'] == model_name]
        if len(model_instance_data) == 0:
            all_models_have_success = False
            break
        
        # Check if this model has at least one successful run
        has_success = False
        for acc_col in acc_cols:
            if acc_col in model_instance_data.columns:
                if model_instance_data[acc_col].iloc[0] == 1:
                    has_success = True
                    break
        
        if not has_success:
            all_models_have_success = False
            break
    
    if all_models_have_success:
        successful_instances.add(instance_id)

print(f"\nFound {len(successful_instances)} instances where all {len(all_models)} models have at least one success")
print(f"Out of {len(df['instance_id'].unique())} total instances")

# Step 2: Calculate total token cost averages for all instances
print("\n" + "=" * 80)
print("Step 2: Calculating total token cost averages for ALL instances...")
print("=" * 80)

all_instances_averages = {}

for model_name in all_models:
    df_model = df[df['model'] == model_name].copy()
    
    # Calculate total token cost (prompt + completion) for all runs
    all_total_tokens = []
    for run_num in range(1, 5):
        prompt_col = f"total_prompt_tokens_run{run_num}"
        completion_col = f"total_completion_tokens_run{run_num}"
        
        for _, row in df_model.iterrows():
            prompt_tokens = row[prompt_col] if pd.notna(row[prompt_col]) else 0
            completion_tokens = row[completion_col] if pd.notna(row[completion_col]) else 0
            total_tokens = float(prompt_tokens) + float(completion_tokens)
            
            if total_tokens > 0:  # Only add if we have valid data
                all_total_tokens.append(total_tokens)
    
    # Calculate average total token cost for this model across all instances
    if len(all_total_tokens) > 0:
        all_instances_averages[model_name] = np.mean(all_total_tokens)
        print(f"  {model_name}: {np.mean(all_total_tokens):.2e} (from {len(all_total_tokens)} runs)")
    else:
        print(f"  {model_name}: No valid data found")

# Step 2b: Calculate total money cost averages for all instances
print("\n" + "=" * 80)
print("Step 2b: Calculating total money cost averages for ALL instances...")
print("=" * 80)

all_instances_money_averages = {}

for model_name in all_models:
    df_model = df[df['model'] == model_name].copy()
    
    # Calculate total money cost for all runs
    all_total_costs = []
    for run_num in range(1, 5):
        cost_col = f"total_cost_run{run_num}"
        
        for _, row in df_model.iterrows():
            if cost_col in row.index:
                cost = row[cost_col] if pd.notna(row[cost_col]) else 0
                cost = float(cost)
                
                if cost > 0:  # Only add if we have valid data
                    all_total_costs.append(cost)
    
    # Calculate average total money cost for this model across all instances
    if len(all_total_costs) > 0:
        all_instances_money_averages[model_name] = np.mean(all_total_costs)
        print(f"  {model_name}: ${np.mean(all_total_costs):.4f} (from {len(all_total_costs)} runs)")
    else:
        print(f"  {model_name}: No valid data found")

# Step 3: Calculate success subset averages using only successful runs
print("\n" + "=" * 80)
print("Step 3: Calculating model averages for SUCCESS SUBSET (using only successful runs)...")
print("=" * 80)

success_subset_averages = {}

# For each model, calculate average total token cost from successful runs only
for model_name in all_models:
    df_model = df[df['model'] == model_name].copy()
    
    # Filter to success subset instances
    df_model_success = df_model[df_model['instance_id'].isin(successful_instances)].copy()
    
    # For each instance, collect token costs from successful runs only
    success_total_tokens = []
    
    for instance_id in successful_instances:
        instance_row = df_model_success[df_model_success['instance_id'] == instance_id]
        if len(instance_row) == 0:
            continue
        
        row = instance_row.iloc[0]
        
        # Check each run and collect token costs from successful runs
        for run_num in range(1, 5):
            acc_col = f"acc_run{run_num}"
            prompt_col = f"total_prompt_tokens_run{run_num}"
            completion_col = f"total_completion_tokens_run{run_num}"
            
            # Check if this run was successful
            if acc_col in row.index and row[acc_col] == 1:
                # Get token costs for this successful run
                prompt_tokens = row[prompt_col] if pd.notna(row[prompt_col]) else 0
                completion_tokens = row[completion_col] if pd.notna(row[completion_col]) else 0
                total_tokens = float(prompt_tokens) + float(completion_tokens)
                
                if total_tokens > 0:  # Only add if we have valid data
                    success_total_tokens.append(total_tokens)
    
    # Calculate average total token cost for this model in success subset
    if len(success_total_tokens) > 0:
        success_subset_averages[model_name] = np.mean(success_total_tokens)
        print(f"  {model_name}: {np.mean(success_total_tokens):.2e} (from {len(success_total_tokens)} successful runs)")
    else:
        print(f"  {model_name}: No successful runs found in success subset")

# Step 3b: Calculate success subset money cost averages using only successful runs
print("\n" + "=" * 80)
print("Step 3b: Calculating model money cost averages for SUCCESS SUBSET (using only successful runs)...")
print("=" * 80)

success_subset_money_averages = {}

# For each model, calculate average total money cost from successful runs only
for model_name in all_models:
    df_model = df[df['model'] == model_name].copy()
    
    # Filter to success subset instances
    df_model_success = df_model[df_model['instance_id'].isin(successful_instances)].copy()
    
    # For each instance, collect money costs from successful runs only
    success_total_costs = []
    
    for instance_id in successful_instances:
        instance_row = df_model_success[df_model_success['instance_id'] == instance_id]
        if len(instance_row) == 0:
            continue
        
        row = instance_row.iloc[0]
        
        # Check each run and collect money costs from successful runs
        for run_num in range(1, 5):
            acc_col = f"acc_run{run_num}"
            cost_col = f"total_cost_run{run_num}"
            
            # Check if this run was successful
            if acc_col in row.index and row[acc_col] == 1:
                # Get money costs for this successful run
                if cost_col in row.index:
                    cost = row[cost_col] if pd.notna(row[cost_col]) else 0
                    cost = float(cost)
                    
                    if cost > 0:  # Only add if we have valid data
                        success_total_costs.append(cost)
    
    # Calculate average total money cost for this model in success subset
    if len(success_total_costs) > 0:
        success_subset_money_averages[model_name] = np.mean(success_total_costs)
        print(f"  {model_name}: ${np.mean(success_total_costs):.4f} (from {len(success_total_costs)} successful runs)")
    else:
        print(f"  {model_name}: No successful runs found in success subset")

# Prepare data for grouped bar plot - use the ordered model list
models = all_models  # Already in the correct order
y_pos = np.arange(len(models))
height = 0.35  # Height of the bars (for horizontal bars)

# Get values for both bars in the correct order
all_instances_values = [all_instances_averages.get(m, 0) for m in models]
success_subset_values = [success_subset_averages.get(m, 0) for m in models]

# Get money cost values for both bars in the correct order
all_instances_money_values = [all_instances_money_averages.get(m, 0) for m in models]
success_subset_money_values = [success_subset_money_averages.get(m, 0) for m in models]

# ===== TOKEN COST PLOT (horizontal) =====
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))

bars1_token = ax1.barh(
    y_pos - height / 2,
    all_instances_values,
    height,
    label='All Instances',
    color='#C5DFF8',
    alpha=0.7,
)
bars2_token = ax1.barh(
    y_pos + height / 2,
    success_subset_values,
    height,
    label=f'Success Subset ({len(successful_instances)} instances)',
    color='#7895CB',
    alpha=0.7,
)

# Set labels and titles with larger fonts
# Remove plot title
ax1.set_title("")
# Remove x-axis label
ax1.set_xlabel("")
# Remove y-axis label but keep the model names
ax1.set_ylabel("")
ax1.set_yticks(y_pos)
ax1.set_yticklabels(models, fontsize=32)

# Set tick label sizes for x-axis
ax1.tick_params(axis='x', labelsize=22)
# Increase y-axis tick label size
ax1.tick_params(axis='y', labelsize=32)

# Remove legend for token cost plot
# ax1.legend(fontsize=24, loc='best')

# Remove right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Add grid
ax1.grid(True, linestyle="--", alpha=0.5, axis='x')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the token cost figure
token_output_path = OUTPUT_DIR / "model_token_usage_comparison.png"
plt.savefig(token_output_path, dpi=300, bbox_inches='tight')
print(f"\nToken cost figure saved to: {token_output_path}")

plt.show()

# ===== MONEY COST PLOT (horizontal) =====
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))

bars1_money = ax2.barh(
    y_pos - height / 2,
    all_instances_money_values,
    height,
    label='All Instances',
    color='#C5DFF8',
    alpha=0.7,
)
bars2_money = ax2.barh(
    y_pos + height / 2,
    success_subset_money_values,
    height,
    label=f'Success Subset ({len(successful_instances)} instances)',
    color='#7895CB',
    alpha=0.7,
)

# Set labels and titles with larger fonts
# Remove plot title
ax2.set_title("")
# Remove x-axis label
ax2.set_xlabel("")
# Remove y-axis label
ax2.set_ylabel("")
ax2.set_yticks(y_pos)
# Remove y-axis tick labels but keep the ticks
ax2.set_yticklabels([""] * len(models))

# Set tick label sizes for x-axis
ax2.tick_params(axis='x', labelsize=22)
# Remove y-axis tick labels
ax2.tick_params(axis='y', labelsize=0, length=0)

# Add legend with larger font, positioned at top
ax2.legend(fontsize=25, loc='upper right')

# Remove right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Add grid
ax2.grid(True, linestyle="--", alpha=0.5, axis='x')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the money cost figure
money_output_path = OUTPUT_DIR / "model_money_cost_comparison.png"
plt.savefig(money_output_path, dpi=300, bbox_inches='tight')
print(f"Money cost figure saved to: {money_output_path}")

plt.show()

# Print summary statistics
print(f"\n" + "=" * 80)
print("Summary Statistics:")
print("=" * 80)
print(f"\nAll Instances Total Token Cost:")
print(f"  Models: {len([v for v in all_instances_values if v > 0])}")
print(f"  Min: {min([v for v in all_instances_values if v > 0]):.2e} ({models[np.argmin([v if v > 0 else np.inf for v in all_instances_values])]})")
print(f"  Max: {max(all_instances_values):.2e} ({models[np.argmax(all_instances_values)]})")
print(f"  Mean: {np.mean([v for v in all_instances_values if v > 0]):.2e}")

print(f"\nSuccess Subset Total Token Cost:")
print(f"  Success subset size: {len(successful_instances)} instances")
print(f"  Models: {len([v for v in success_subset_values if v > 0])}")
print(f"  Min: {min([v for v in success_subset_values if v > 0]):.2e} ({models[np.argmin([v if v > 0 else np.inf for v in success_subset_values])]})")
print(f"  Max: {max(success_subset_values):.2e} ({models[np.argmax(success_subset_values)]})")
print(f"  Mean: {np.mean([v for v in success_subset_values if v > 0]):.2e}")

print(f"\nAll Instances Total Money Cost:")
print(f"  Models: {len([v for v in all_instances_money_values if v > 0])}")
print(f"  Min: ${min([v for v in all_instances_money_values if v > 0]):.4f} ({models[np.argmin([v if v > 0 else np.inf for v in all_instances_money_values])]})")
print(f"  Max: ${max(all_instances_money_values):.4f} ({models[np.argmax(all_instances_money_values)]})")
print(f"  Mean: ${np.mean([v for v in all_instances_money_values if v > 0]):.4f}")

print(f"\nSuccess Subset Total Money Cost:")
print(f"  Success subset size: {len(successful_instances)} instances")
print(f"  Models: {len([v for v in success_subset_money_values if v > 0])}")
print(f"  Min: ${min([v for v in success_subset_money_values if v > 0]):.4f} ({models[np.argmin([v if v > 0 else np.inf for v in success_subset_money_values])]})")
print(f"  Max: ${max(success_subset_money_values):.4f} ({models[np.argmax(success_subset_money_values)]})")
print(f"  Mean: ${np.mean([v for v in success_subset_money_values if v > 0]):.4f}")


print("\n" + "=" * 80)
print("Model Token Usage Comparison complete!")
print("=" * 80)
